"""Focused app-level tests for the UEP workflow integration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import UUID

import pytest
from PySide6.QtWidgets import QDialog

import mufinder.app.main_window as main_window_module
import mufinder.app.workflow_controllers.uep as uep_controller_module
from mufinder.app.main_window import MainWindow
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.services.project_service import ProjectService
from mufinder.services.uep_results import uep_batch_search, uep_cluster_source_rows
from tests.support.uep_results import store_search
from tests.support.uep_sources import (
    completed_uep_job_with_grid as _completed_uep_job_with_grid,
    uep_fixture_text as _fixture_text,
    uep_host_structure as _host_structure,
)


def test_main_window_workflow_tree_contains_uep_stage(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_route.mufinder", "UEPRoute")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._refresh_ui_after_project_change()

    assert window._project_navigator.has_route("uep")


def test_main_window_on_uep_configured_registers_definition(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_configured.mufinder", "UEPConfigured")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["uep_form"] = {"selected_source_structure_id": str(host.id)}
    window._document = doc

    window._uep_controller.configure(
        UepConfig(
            name_prefix="uep",
            rows=(UepSourceRow(label="row_1", parameter_values={}),),
        )
    )

    definitions = [
        definition
        for definition in doc.calculations.values()
        if definition.workflow_stage == WorkflowStage.UEP
    ]
    assert len(definitions) == 1
    assert definitions[0].calc_type == "castep_uep_source"
    assert definitions[0].input_structure_ids == [host.id]


def test_main_window_local_finish_uep_source_marks_density_available(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_finish.mufinder", "UEPFinish")
    job = service.add_job(
        doc,
        label="uep_row1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / f"{job.label}.castep").write_text("castep", encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text("density", encoding="utf-8")

    parsed = CastepResult(
        completed=True,
        total_energy=-20.5,
        wall_time=14.0,
        warnings=(),
    )
    window._jobs._on_local_job_finished(job.id, parsed)

    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["density_available"] is True
    assert result.data["total_energy"] == -20.5
    assert any(path.endswith(f"{job.label}.den_fmt") for path in result.output_files)


def test_main_window_open_uep_analysis_launches_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_launch.mufinder", "UEPLaunch")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {str(job.id): {"current_tab": 2}}
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[run.id] = run

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.state_changed = _SignalStub()
            self.search_changed = _SignalStub()
            self.destroyed = _SignalStub()
            captured["instance"] = self

        def setModal(self, modal: bool) -> None:
            captured["modal"] = modal

        def show(self) -> None:
            captured["shown"] = True

        def raise_(self) -> None:
            captured["raised"] = True

        def activateWindow(self) -> None:
            captured["activated"] = True

        def close(self) -> bool:
            captured["closed"] = True
            return True

        def state_snapshot(self) -> dict[str, object]:
            return {"current_tab": 2}

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeDialog)
    window._uep_controller.open_analysis("row_1")

    assert captured["source_job_label"] == job.label
    assert captured["source_structure_id"] == host.id
    assert captured["source_structure"] is host
    assert captured["initial_state"] == {"current_tab": 2}
    assert captured["initial_search"] is None
    assert "initial_potential_grid" not in captured
    assert captured["modal"] is False
    assert captured["shown"] is True
    assert captured["raised"] is True
    assert captured["activated"] is True
    assert window._uep_controller.analysis_dialogs[job.id] is captured["instance"]


def _capture_uep_analysis_dialog(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.state_changed = _SignalStub()
            self.search_changed = _SignalStub()
            self.destroyed = _SignalStub()

        def setModal(self, _modal: bool) -> None:
            return None

        def show(self) -> None:
            return None

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def close(self) -> bool:
            return True

        def state_snapshot(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeDialog)
    return captured


def test_main_window_open_uep_analysis_hands_the_window_the_stored_search(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    doc, job, run, _grid = _completed_uep_job_with_grid(window, tmp_path)
    record = store_search(
        doc, job.id, results=[{"classical_energy_ev": -1.0}], log="Search log", settings={}
    )
    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {str(job.id): {"gw_factor": 5.0}}
    captured = _capture_uep_analysis_dialog(monkeypatch)

    window._uep_controller.open_analysis("row_1")

    assert captured["initial_state"] == {"gw_factor": 5.0}
    assert captured["initial_search"] == {
        "settings": {},
        "results": record.data["results"],
        "grouping": None,
        "log": "Search log",
    }
    assert "initial_potential_grid" not in captured


#: Keys of the point-probe, line and plane pages, which older projects saved in each UEP
#: analysis state. Nothing reads them.
_RETIRED_UEP_PAGE_STATE = {
    "probe_point": [0.25, 0.25, 0.25],
    "latest_probe": {"position_ang": [1.0, 1.0, 1.0], "potential_total_ev": -3.2},
    "probe_output": "Total potential (eV): -3.20000000",
    "line_start": [0.0, 0.0, 0.0],
    "line_end": [0.5, 0.5, 0.5],
    "line_points": 11,
    "line_sample": {"distance_ang": [0.0, 1.0], "potential_total_ev": [0.1, 0.2]},
    "line_data": "distance_A,total_eV,electronic_eV,ionic_eV",
    "plane_origin": [0.0, 0.0, 0.0],
    "plane_x": [0.5, 0.0, 0.0],
    "plane_y": [0.0, 0.5, 0.0],
    "plane_points_x": 61,
    "plane_points_y": 61,
    "plane_sample": {"potential_total_ev": [[0.1, 0.2], [0.3, 0.4]]},
    "plane_data": "Grid shape: 2 x 2",
}


def test_a_project_saved_with_the_retired_uep_page_keys_opens_and_saves_without_them(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    doc, job, _run, _grid = _completed_uep_job_with_grid(window, tmp_path)
    service = ProjectService()
    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {
        str(job.id): {"gw_factor": 5.0, "batch_count": 7, **_RETIRED_UEP_PAGE_STATE}
    }
    service.save_project(doc)

    window._document = service.load_project(doc.project_root)
    window._uep_controller.open_analysis("row_1")
    dialog = window._uep_controller.analysis_dialogs[job.id]
    snapshot = dialog.state_snapshot()

    assert dialog._batch_count.value() == 7
    assert set(_RETIRED_UEP_PAGE_STATE).isdisjoint(snapshot)
    dialog._emit_state_changed()
    dialog.flush_pending_state()
    service.save_project(window._document)
    dialog.close()

    saved = service.load_project(doc.project_root).ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert set(_RETIRED_UEP_PAGE_STATE).isdisjoint(saved[str(job.id)])
    assert saved[str(job.id)]["batch_count"] == 7
    assert saved[str(job.id)]["gw_factor"] == pytest.approx(5.0)


def test_main_window_open_uep_analysis_does_not_request_snapshot_after_dialog_show(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "uep_no_post_exec_snapshot.mufinder", "UEPNoPostExecSnapshot"
    )
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")

    class _FakeDialog:
        def __init__(self, **_kwargs) -> None:
            self.state_changed = _SignalStub()
            self.search_changed = _SignalStub()
            self.destroyed = _SignalStub()

        def setModal(self, _modal: bool) -> None:
            return None

        def show(self) -> None:
            return None

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def close(self) -> bool:
            return True

        def state_snapshot(self) -> dict[str, object]:
            raise AssertionError("state_snapshot should not be requested after show() returns")

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeDialog)

    window._uep_controller.open_analysis("row_1")


def test_main_window_open_uep_analysis_reuses_existing_window(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_reuse_window.mufinder", "UEPReuseWindow")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")

    captured = {"constructed": 0, "shown": 0, "raised": 0, "activated": 0}

    class _FakeDialog:
        def __init__(self, **_kwargs) -> None:
            captured["constructed"] += 1
            self.state_changed = _SignalStub()
            self.search_changed = _SignalStub()
            self.destroyed = _SignalStub()

        def setModal(self, _modal: bool) -> None:
            return None

        def show(self) -> None:
            captured["shown"] += 1

        def raise_(self) -> None:
            captured["raised"] += 1

        def activateWindow(self) -> None:
            captured["activated"] += 1

        def close(self) -> bool:
            return True

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeDialog)

    window._uep_controller.open_analysis("row_1")
    window._uep_controller.open_analysis("row_1")

    assert captured["constructed"] == 1
    assert captured["shown"] == 2
    assert captured["raised"] == 2
    assert captured["activated"] == 2


def test_main_window_open_uep_analysis_recreates_closed_window(
    qapp, main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "uep_recreate_closed_window.mufinder", "UEPRecreateClosedWindow"
    )
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")

    constructed: list[QDialog] = []

    class _FakeDialog(QDialog):
        def __init__(self, **kwargs) -> None:
            super().__init__(kwargs.get("parent"))
            self.state_changed = _SignalStub()
            self.search_changed = _SignalStub()
            constructed.append(self)

        def state_snapshot(self) -> dict[str, object]:
            return {"current_tab": 0}

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeDialog)

    window._uep_controller.open_analysis("row_1")
    first = constructed[0]
    first.close()
    qapp.processEvents()
    window._uep_controller.open_analysis("row_1")

    assert len(constructed) == 2
    assert window._uep_controller.analysis_dialogs[job.id] is constructed[1]


def test_main_window_close_saves_before_closing_open_uep_dialogs(
    qapp, main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_close_before_save.mufinder", "UEPCloseBeforeSave")
    window._document = doc
    window.show()

    events: list[str] = []

    class _FakeUepDialog(QDialog):
        def close(self) -> bool:  # type: ignore[override]
            events.append("dialog.close")
            return super().close()

    monkeypatch.setattr(uep_controller_module, "UepAnalysisDialog", _FakeUepDialog)

    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        events.append("save_project")
        original_save(document)

    monkeypatch.setattr(window._project_service, "save_project", _save_project)

    dialog = _FakeUepDialog(window)
    dialog.show()
    qapp.processEvents()

    window.close()

    assert events[:2] == ["save_project", "dialog.close"]


def test_main_window_uep_analysis_state_changed_updates_document_but_not_disk_until_save(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_state.mufinder", "UEPState")
    window._document = doc

    job_id = UUID("11111111-1111-1111-1111-111111111111")
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")
    window._uep_controller.analysis_state_changed(
        job_id,
        {
            "current_tab": 3,
            "batch_count": 7,
            "batch_selected_keys": ["cluster:1"],
        },
    )

    stored = doc.ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert stored[str(job_id)]["current_tab"] == 3
    assert stored[str(job_id)]["batch_count"] == 7
    assert stored[str(job_id)]["batch_selected_keys"] == ["cluster:1"]
    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text

    service.save_project(doc)

    reloaded = service.load_project(doc.project_root)
    persisted = reloaded.ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert persisted[str(job_id)]["current_tab"] == 3
    assert persisted[str(job_id)]["batch_count"] == 7
    assert persisted[str(job_id)]["batch_selected_keys"] == ["cluster:1"]


def test_main_window_uep_analysis_state_change_keeps_dialog_visible_without_table_refresh(
    qapp,
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_isosurface_state.mufinder", "UEPIsosurfaceState")
    window._document = doc

    job_id = UUID("11111111-1111-1111-1111-111111111111")
    cluster_state = {
        "batch_results": [{"optimized_fractional_position": [0.1, 0.2, 0.3]}],
        "batch_summary_data": {"clusters": [{"cluster_index": 1, "representative_index": 0}]},
        "isosurface_layers": [{"label": "Isosurface 1", "level_ev": -0.1}],
    }
    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {str(job_id): dict(cluster_state)}
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        window, "_refresh_muon_generation_table", lambda: refresh_calls.append("refresh")
    )

    class _FakeDialog:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def isVisible(self) -> bool:
            return True

        def show(self) -> None:
            self.calls.append("show")

        def raise_(self) -> None:
            self.calls.append("raise")

        def activateWindow(self) -> None:
            self.calls.append("activate")

    dialog = _FakeDialog()
    window._uep_controller.analysis_dialogs[job_id] = dialog

    updated_state = dict(cluster_state)
    updated_state["isosurface_layers"] = [
        {"label": "Isosurface 1", "level_ev": -0.1},
        {"label": "Isosurface 2", "level_ev": 0.2},
    ]
    window._uep_controller.analysis_state_changed(job_id, updated_state)
    qapp.processEvents()

    assert refresh_calls == []
    assert dialog.calls == ["show", "raise", "activate"]
    assert (
        doc.ui_state[window._UEP_ANALYSIS_STATES_KEY][str(job_id)]["isosurface_layers"]
        == updated_state["isosurface_layers"]
    )


def test_a_stored_search_asks_for_an_autosave_and_a_view_change_does_not(
    qapp,
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_batch_state.mufinder", "UEPBatchState")
    job = service.add_job(
        doc, label="uep_row_1", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source"
    )
    window._document = doc
    flushes: list[str] = []
    monkeypatch.setattr(
        window,
        "_request_dashboard_flush",
        lambda *, failure_message: flushes.append(failure_message),
    )

    window._uep_controller.analysis_state_changed(job.id, {"current_tab": 1})
    assert flushes == []
    search = {"results": [{"optimized_fractional_position": [0.1, 0.2, 0.3]}], "log": ""}
    window._uep_controller.search_changed(job.id, search)
    window._uep_controller.search_changed(job.id, search)
    qapp.processEvents()

    assert flushes == ["The UEP search was kept, but autosave failed."]


def test_a_finished_search_is_on_disk_after_the_next_autosave_without_save_project(
    qapp, main_window, tmp_path: Path
) -> None:
    window = main_window
    doc, job, run, _grid = _completed_uep_job_with_grid(window, tmp_path)
    service = ProjectService()
    service.save_project(doc)
    search = {
        "settings": {"gw_factor": 5.0, "batch_count": 1},
        "results": [{"classical_energy_ev": -0.7, "optimized_position_ang": [0.5, 1.0, 1.5]}],
        "grouping": {
            "threshold_ang": 0.25,
            "clustering_method": "symmetry_distance",
            "clusters": [{"cluster_index": 1, "size": 1, "representative_index": 0}],
        },
        "log": "Requested random starts: 1",
    }

    window._uep_controller.search_changed(job.id, search)
    window._flush_dashboard()

    saved = service.load_project(doc.project_root)
    stored = uep_batch_search(saved, job.id)
    assert stored is not None
    assert stored.payload() == search
    assert saved.results[stored.result_id].calculation_run_id == run.id
    # The window's view is not part of it.
    assert str(job.id) not in saved.ui_state.get(window._UEP_ANALYSIS_STATES_KEY, {})


def test_a_discarded_search_leaves_the_project_at_the_next_autosave(
    qapp, main_window, tmp_path: Path
) -> None:
    window = main_window
    doc, job, _run, _grid = _completed_uep_job_with_grid(window, tmp_path)
    store_search(doc, job.id, results=[{"classical_energy_ev": -0.7}])
    ProjectService().save_project(doc)

    window._uep_controller.search_changed(job.id, None)
    window._flush_dashboard()

    assert uep_batch_search(ProjectService().load_project(doc.project_root), job.id) is None


def test_main_window_exposes_uep_cluster_rows_for_generation(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "uep_cluster_generation.mufinder", "UEPClusterGeneration"
    )
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    run = CalculationRun(
        definition_id=definition.id, job_id=job.id, run_index=1, status=JobStatus.COMPLETED
    )
    doc.runs[run.id] = run
    store_search(
        doc,
        job.id,
        results=[
            {
                "start_position_ang": [0.0, 0.0, 0.0],
                "optimized_position_ang": [0.5, 1.0, 1.5],
                "optimized_fractional_position": [0.1, 0.2, 0.3],
                "classical_energy_ev": -0.7,
                "zero_point_energy_ev": 0.2,
                "total_energy_ev": -0.5,
                "success": True,
                "message": "ok",
                "iterations": 3,
                "classification": {"label": "minimum"},
            }
        ],
        grouping={
            "threshold_ang": 0.3,
            "clustering_method": "hier",
            "clustering_hier_t": 0.3,
            "clustering_kmeans_k": None,
            "clustering_linkage_method": "single",
            "feature_mode": "pymuon",
            "clusters": [
                {
                    "cluster_index": 1,
                    "size": 1,
                    "representative_index": 0,
                    "representative_position_ang": [0.5, 1.0, 1.5],
                    "representative_fractional_position": [0.1, 0.2, 0.3],
                    "minimum_energy_ev": -0.5,
                    "average_energy_ev": -0.5,
                    "standard_deviation_ev": 0.0,
                    "member_indexes": [0],
                }
            ],
        },
    )
    window._document = doc

    rows = uep_cluster_source_rows(doc, {job.id})

    assert len(rows) == 1
    representative = rows[0]
    assert representative["generation_method"] == "uep_cluster"
    assert representative["selection_kind"] == "cluster"
    assert representative["source_structure_id"] == str(host.id)
    assert representative["properties"]["classification"] == "minimum"

    window._site_generation_controller.generate("uep_clusters", {"selections": [representative]})

    assert len(doc.sites) == 1
    site = next(iter(doc.sites.values()))
    assert site.generation_method == "uep_cluster"
    assert site.source_structure_id == host.id
    assert site.properties["classification"] == "minimum"


def test_a_generation_row_reads_the_group_and_records_its_member(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_group_row.mufinder", "UEPGroupRow")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    job = service.add_job(
        doc, label="row_1", workflow_stage=WorkflowStage.UEP, calc_type="uep_source"
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)
    store_search(
        doc,
        job.id,
        results=[
            {
                "optimized_position_ang": [9.0, 9.0, 9.0],
                "optimized_fractional_position": [0.99999999, 0.5, 0.75],
                "classical_energy_ev": -0.7,
                "classification": {"label": "minimum"},
            }
        ],
        grouping={
            "threshold_ang": 0.25,
            "energy_tolerance_ev": 0.001,
            "clustering_method": "symmetry_distance",
            "clusters": [
                {
                    "cluster_index": 1,
                    "size": 12,
                    "representative_index": 0,
                    "representative_position_ang": [0.0, 2.5, 1.25],
                    "representative_fractional_position": [0.0, 0.5, 0.25],
                    "member_indexes": [0],
                }
            ],
        },
    )
    window._document = doc

    (row,) = uep_cluster_source_rows(doc, window._uep_controller.active_job_ids())

    assert row["fractional_position"] == [0.0, 0.5, 0.25]
    assert row["cartesian_position"] == [0.0, 2.5, 1.25]
    assert row["cluster_size"] == 12
    assert row["label"] == "UEP Site 1"
    assert row["properties"]["member_index"] == 0
    assert row["properties"]["threshold_ang"] == 0.25
    assert row["properties"]["energy_tolerance_ev"] == 0.001
    assert row["properties"]["clustering_method"] == "symmetry_distance"


def test_saving_writes_a_grouping_tolerance_that_has_not_settled(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_pending.mufinder", "UEPPending")
    window._document = doc
    job_id = UUID("11111111-1111-1111-1111-111111111111")

    class _PendingDialog:
        """A window whose regroup has not been emitted yet."""

        def flush_pending_state(self) -> None:
            window._uep_controller.analysis_state_changed(job_id, {"current_tab": 7})

    window._uep_controller.analysis_dialogs[job_id] = _PendingDialog()
    try:
        window._save_project()
    finally:
        window._uep_controller.analysis_dialogs.clear()

    saved = json.loads(doc.project_root.joinpath("state.json").read_text(encoding="utf-8"))
    assert saved[window._UEP_ANALYSIS_STATES_KEY][str(job_id)]["current_tab"] == 7


def test_main_window_ignores_stale_uep_cluster_states_for_generation(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "uep_stale_cluster_generation.mufinder", "UEPStaleClusterGeneration"
    )
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    stale_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep_old",
    )
    stale_job = service.add_job(
        doc,
        label="uep_old_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    stale_definition.parameters["job_ids_by_label"] = {"row_1": str(stale_job.id)}
    stale_run = CalculationRun(
        definition_id=stale_definition.id,
        job_id=stale_job.id,
        run_index=1,
        status=JobStatus.COMPLETED,
    )
    doc.runs[stale_run.id] = stale_run

    active_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    active_job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    active_definition.parameters["job_ids_by_label"] = {"row_1": str(active_job.id)}
    active_run = CalculationRun(
        definition_id=active_definition.id,
        job_id=active_job.id,
        run_index=1,
        status=JobStatus.COMPLETED,
    )
    doc.runs[active_run.id] = active_run
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(active_definition.id)

    def store(job_id, cluster_index: int, fractional_x: float) -> None:
        store_search(
            doc,
            job_id,
            results=[
                {
                    "optimized_position_ang": [fractional_x, 1.0, 1.5],
                    "optimized_fractional_position": [fractional_x, 0.2, 0.3],
                    "classical_energy_ev": -0.7,
                    "classification": {"label": "minimum"},
                }
            ],
            grouping={
                "threshold_ang": 0.3,
                "clustering_method": "hier",
                "clusters": [
                    {
                        "cluster_index": cluster_index,
                        "size": 1,
                        "representative_index": 0,
                        "member_indexes": [0],
                    }
                ],
            },
        )

    store(stale_job.id, 99, 0.9)
    store(active_job.id, 1, 0.1)
    window._document = doc

    rows = uep_cluster_source_rows(doc, window._uep_controller.active_job_ids())

    assert rows
    assert {row["source_job_id"] for row in rows} == {str(active_job.id)}
    assert {row["cluster_index"] for row in rows} == {1}


def _charged_host_output(text: str) -> str:
    return text.replace(
        " net charge of system                           :  0.000",
        " net charge of system                           : +1.000",
    )


def _output_without_pseudopotential_reports(text: str) -> str:
    return text.replace("Pseudopotential Report", "Report")


@pytest.mark.parametrize(
    ("castep_name", "den_fmt_name", "edit_output", "refusal"),
    [
        ("Si2.castep", "Si2.den_fmt", _charged_host_output, "reports total charge 1"),
        ("Si2-charged.castep", "Si2-charged.den_fmt", None, "formatted density integrates"),
        (
            "Si2.castep",
            "Si2.den_fmt",
            _output_without_pseudopotential_reports,
            "Missing pseudopotential metadata",
        ),
    ],
    ids=["charged-host", "density-not-ionic-charge", "no-pseudopotential"],
)
def test_a_refused_uep_source_leaves_no_window_behind_and_the_main_window_closes(
    qapp,
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    castep_name: str,
    den_fmt_name: str,
    edit_output,
    refusal: str,
) -> None:
    from mufinder.app.uep_wizard import UepAnalysisDialog

    window = main_window
    _doc, job, _run, _grid = _completed_uep_job_with_grid(window, tmp_path)
    output_dir = window._document.project_root / "files" / "uep" / job.label
    castep_text = _fixture_text(castep_name)
    if edit_output is not None:
        castep_text = edit_output(castep_text)
        assert castep_text != _fixture_text(castep_name)
    (output_dir / f"{job.label}.castep").write_text(castep_text, encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text(den_fmt_name), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, text, *_args: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        main_window_module.workflow_table_utils, "confirm_choice", lambda *_a, **_k: "discard"
    )
    # If the close fails, the window stays open; the per-test clean-up must not then
    # stop at the save question, which this test answers only while it runs.
    request.addfinalizer(lambda: setattr(window, "_shutting_down", True))
    window.show()
    qapp.processEvents()

    window._uep_controller.open_analysis("row_1")
    qapp.processEvents()

    assert [title for title, _text in warnings] == ["UEP analysis unavailable"]
    assert refusal in warnings[0][1]
    assert window.findChildren(UepAnalysisDialog) == []
    assert job.id not in window._uep_controller.analysis_dialogs

    # Qt reports an exception raised in a Python override through sys.excepthook.
    raised: list[BaseException] = []
    monkeypatch.setattr(sys, "excepthook", lambda _type, value, _tb: raised.append(value))
    window.close()
    qapp.processEvents()

    assert raised == []
    assert not window.isVisible()


def test_a_dashboard_flush_with_a_large_grid_uep_job_serialises_no_grid_mesh_or_sample(
    qapp, main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 96 x 96 x 64 density, the size of a real CASTEP fine grid, as the perf harness builds it.

    The analysis window shows its potential with a surface and a plane, and holds a
    search. Nothing of the grid, the mesh or the plane's sample reaches the document,
    so a flush stays cheap however fine the grid.
    """

    import statistics
    import sys
    import time

    from mufinder.app import uep_wizard
    from mufinder.persistence.codec import to_jsonable
    from mufinder.science.uep import plane_from_fractional_anchor_and_miller
    from tests.support.symmetry_hosts import cof2_host

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from docs_screenshots.synthetic import den_fmt_text, uep_castep_text

    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr(uep_wizard._IsosurfaceGridThread, "start", _start_inline)
    monkeypatch.setattr(uep_wizard._IsosurfaceMeshThread, "start", _start_inline)
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_fine_grid.mufinder", "UEPFineGrid")
    host = cof2_host()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc, label="uep_row_1", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source"
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    run = CalculationRun(job_id=job.id, run_index=1)
    doc.runs[run.id] = run
    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True)
    cell = {
        "lattice": host.lattice_vectors,
        "species": host.species,
        "fractional": host.fractional_positions,
    }
    (output_dir / f"{job.label}.den_fmt").write_text(
        den_fmt_text(
            **cell,
            grid=(96, 96, 64),
            shells={"Co": ((8.0, 0.22), (7.0, 0.42)), "F": ((2.0, 0.25), (6.0, 0.60))},
        ),
        encoding="utf-8",
    )
    (output_dir / f"{job.label}.castep").write_text(
        uep_castep_text(
            **cell,
            pseudopotentials={"Co": (17.0, 1.8), "F": (7.0, 1.4)},
            energy=-5621.8722,
            wall_time=1843.2,
        ),
        encoding="utf-8",
    )
    window._document = doc
    window._uep_controller.open_analysis("row_1")
    dialog = window._uep_controller.analysis_dialogs[job.id]
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()
    dialog._isosurface_level_input.setValue(1.0)
    dialog._isosurface_level_input.editingFinished.emit()
    dialog._slice_planes = [
        uep_wizard._SlicePlaneState(
            label="Plane 1",
            plane=plane_from_fractional_anchor_and_miller(
                anchor_fractional=(0.0, 0.0, 0.5),
                miller_indices=(0.0, 0.0, 1.0),
                lattice_vectors=dialog._source.structure.lattice_vectors,
                points_u=161,
                points_v=161,
            ),
        )
    ]
    dialog._resample_all_slice_planes()
    dialog._emit_state_changed()
    assert dialog._potential_grid_cache.grid == (96, 96, 64)
    assert dialog._isosurface_layers[0].mesh is not None
    window._uep_controller.search_changed(
        job.id, {"results": [{"classical_energy_ev": -1.0}], "grouping": None, "log": ""}
    )
    window._uep_controller.flush_analysis_dialogs()
    service.save_project(doc)

    saved = json.dumps([to_jsonable(doc.ui_state), *(to_jsonable(r) for r in doc.results.values())])
    for heavy in ("potential_total_ev", "vertices_ang", "faces", "sample", "x_coordinates_ang"):
        assert heavy not in saved
    assert len(saved) < 50_000
    samples = []
    for _ in range(5):
        window._dashboard_flush_pending = True
        started = time.perf_counter()
        window._flush_dashboard()
        samples.append((time.perf_counter() - started) * 1000.0)
    dialog.close()
    # The issue measured 330 ms on this project before; a generous bound for slow machines.
    assert statistics.median(samples) < 50.0


class _SignalStub:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


def _project_with_a_relaxed_source(
    tmp_path: Path, *, chosen_id: str | None = None, name: str = "UEPSource"
):
    """Save a project with a host and its pristine-relaxed structure.

    ``chosen_id`` is saved as the UEP page's source choice. Returns the project
    root, the host and the relaxed structure.
    """

    service = ProjectService()
    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    relaxed = _host_structure(
        label="Relaxed host", origin="pristine_relaxation", parent_structure_id=host.id
    )
    service.add_structure(doc, relaxed)
    if chosen_id is not None:
        doc.ui_state["uep_form"] = {"selected_source_structure_id": chosen_id}
    service.save_project(doc)
    return doc.project_root, host, relaxed


def _saved_uep_source_choice(project_root: Path) -> object:
    saved = ProjectService().load_project(project_root).ui_state.get("uep_form", {})
    return saved.get("selected_source_structure_id")


def _assert_uep_page_uses(window: MainWindow, structure) -> None:
    """The picker, the summary line and the structure Run builds on are ``structure``."""

    view = window._uep_view
    assert view.selected_source_structure_id() == str(structure.id)
    assert view._source_structure_summary.text() == f"Using source structure: {structure.label}"
    # Run configures the page's rows first, on the selected source structure.
    view._add_row()
    config = view._build_config_or_warn()
    assert config is not None
    window._uep_controller.configure(config)
    assert window._uep_controller.active_definition().input_structure_ids == [structure.id]


def _choose_uep_source(window: MainWindow, structure) -> None:
    combo = window._uep_view._source_structure_combo
    combo.setCurrentIndex(combo.findData(str(structure.id)))


def test_the_chosen_uep_source_structure_survives_save_and_reopen(
    main_window, tmp_path: Path
) -> None:
    project_root, _host, relaxed = _project_with_a_relaxed_source(tmp_path)
    assert main_window.open_project(project_root)
    _choose_uep_source(main_window, relaxed)
    main_window._save_project()
    assert _saved_uep_source_choice(project_root) == str(relaxed.id)

    # Reopened as after a restart: a new window, whose picker has never been filled.
    window = MainWindow()
    assert window.open_project(project_root)

    stored = window._document.ui_state["uep_form"]["selected_source_structure_id"]
    assert stored == str(relaxed.id)
    window._save_project()
    assert _saved_uep_source_choice(project_root) == str(relaxed.id)
    _assert_uep_page_uses(window, relaxed)


def test_a_uep_source_structure_gone_on_reopen_falls_back_to_the_host(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    project_root, host, _relaxed = _project_with_a_relaxed_source(
        tmp_path, chosen_id="00000000-0000-4000-8000-000000000000"
    )

    assert window.open_project(project_root)

    _assert_uep_page_uses(window, host)


def test_a_uep_source_of_a_replaced_host_falls_back_to_the_new_host(
    main_window, tmp_path: Path
) -> None:
    project_root, _host, relaxed = _project_with_a_relaxed_source(tmp_path)
    assert main_window.open_project(project_root)
    _choose_uep_source(main_window, relaxed)
    service = main_window._project_service
    new_host = _host_structure(label="New host", lattice=6.0, origin="imported")
    service.add_structure(main_window._document, new_host)
    service.set_active_host_structure(main_window._document, new_host.id)
    main_window._save_project()

    window = MainWindow()
    assert window.open_project(project_root)

    _assert_uep_page_uses(window, new_host)


def test_a_project_opened_after_another_never_saves_the_other_projects_uep_source(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    root_a, _host_a, relaxed_a = _project_with_a_relaxed_source(tmp_path, name="A")
    root_b, host_b, _relaxed_b = _project_with_a_relaxed_source(tmp_path, name="B")
    assert window.open_project(root_a)
    _choose_uep_source(window, relaxed_a)
    window._save_project()

    assert window.open_project(root_b)
    window._save_project()

    assert _saved_uep_source_choice(root_b) in (None, str(host_b.id))
    _assert_uep_page_uses(window, host_b)


def test_a_project_that_never_chose_a_uep_source_stays_on_the_host(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    project_root, host, _relaxed = _project_with_a_relaxed_source(tmp_path)

    assert window.open_project(project_root)

    _assert_uep_page_uses(window, host)
