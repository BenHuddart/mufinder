"""Focused app-level tests for the UEP workflow integration."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Result, Structure
from mufinder.science.uep import UepField, load_uep_source, potential_grid
from mufinder.services.project_service import ProjectService


def _find_route(window: MainWindow, route: str):
    def visit(item):
        if item.data(0, 0x0100) == route:  # Qt.UserRole
            return item
        for index in range(item.childCount()):
            child = item.child(index)
            found = visit(child)
            if found is not None:
                return found
        return None

    root = window._project_tree.topLevelItem(0)
    return None if root is None else visit(root)


def _host_structure() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"atomic_spins": [2.0, -2.0]},
    )


def test_main_window_workflow_tree_contains_uep_stage(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_route.mufinder", "UEPRoute")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._refresh_ui_after_project_change()

    assert _find_route(window, "uep") is not None
    window.close()


def test_main_window_on_uep_configured_registers_definition(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_configured.mufinder", "UEPConfigured")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["uep_form"] = {"selected_source_structure_id": str(host.id)}
    window._document = doc

    window._on_uep_configured(
        UepConfig(
            name_prefix="uep",
            rows=(UepSourceRow(label="row_1", parameter_values={}),),
        )
    )

    definitions = [definition for definition in doc.calculations.values() if definition.workflow_stage == WorkflowStage.UEP]
    assert len(definitions) == 1
    assert definitions[0].calc_type == "castep_uep_source"
    assert definitions[0].input_structure_ids == [host.id]
    window.close()


def test_main_window_deserialize_uep_config_normalizes_legacy_prefix(qapp) -> None:
    window = MainWindow()

    config = window._deserialize_uep_config({"name_prefix": "uep_source", "rows": []})

    assert config is not None
    assert config.name_prefix == "uep"
    window.close()


def test_main_window_local_finish_uep_source_marks_density_available(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_finish.mufinder", "UEPFinish")
    job = service.add_job(
        doc,
        label="uep_row1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
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
    window._on_local_job_finished(job.id, parsed)

    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["density_available"] is True
    assert result.data["total_energy"] == -20.5
    assert any(path.endswith(f"{job.label}.den_fmt") for path in result.output_files)
    window.close()


def test_main_window_open_uep_analysis_launches_dialog(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_launch.mufinder", "UEPLaunch")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
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
            self.potential_grid_ready = _SignalStub()
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

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)
    window._on_open_uep_analysis_requested("row_1")

    assert captured["source_job_label"] == job.label
    assert captured["source_structure_id"] == host.id
    assert captured["source_structure"] is host
    assert captured["initial_state"] == {"current_tab": 2}
    assert captured["initial_potential_grid"] is None
    assert captured["modal"] is False
    assert captured["shown"] is True
    assert captured["raised"] is True
    assert captured["activated"] is True
    assert window._uep_analysis_dialogs[job.id] is captured["instance"]
    window.close()


def test_main_window_open_uep_analysis_loads_persisted_potential_grid(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_grid_launch.mufinder", "UEPGridLaunch")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
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
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[run.id] = run
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")

    grid = potential_grid(
        UepField(
            load_uep_source(
                den_fmt_path=output_dir / f"{job.label}.den_fmt",
                castep_path=output_dir / f"{job.label}.castep",
                cell_path=output_dir / f"{job.label}.cell",
            )
        )
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=window._UEP_POTENTIAL_GRID_RESULT_TYPE,
        data=main_window_module.serialize_potential_grid(grid),
    )
    doc.results[result.id] = result

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.state_changed = _SignalStub()
            self.potential_grid_ready = _SignalStub()
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

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)
    window._on_open_uep_analysis_requested("row_1")

    assert captured["initial_potential_grid"] is not None
    window.close()


def test_main_window_open_uep_analysis_does_not_request_snapshot_after_dialog_show(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_no_post_exec_snapshot.mufinder", "UEPNoPostExecSnapshot")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
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
            self.potential_grid_ready = _SignalStub()
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

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)

    window._on_open_uep_analysis_requested("row_1")

    window.close()


def test_main_window_open_uep_analysis_reuses_existing_window(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_reuse_window.mufinder", "UEPReuseWindow")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
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
            self.potential_grid_ready = _SignalStub()
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

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)

    window._on_open_uep_analysis_requested("row_1")
    window._on_open_uep_analysis_requested("row_1")

    assert captured["constructed"] == 1
    assert captured["shown"] == 2
    assert captured["raised"] == 2
    assert captured["activated"] == 2
    window.close()


def test_main_window_close_closes_open_uep_dialogs_before_saving(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_close_before_save.mufinder", "UEPCloseBeforeSave")
    window._document = doc
    window.show()

    events: list[str] = []

    class _FakeUepDialog(QDialog):
        def close(self) -> bool:  # type: ignore[override]
            events.append("dialog.close")
            return super().close()

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeUepDialog)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        events.append("save_project")
        original_save(document)

    monkeypatch.setattr(window._project_service, "save_project", _save_project)

    dialog = _FakeUepDialog(window)
    dialog.show()
    qapp.processEvents()

    window.close()

    assert events[:2] == ["dialog.close", "save_project"]


def test_main_window_uep_analysis_state_changed_updates_document_but_not_disk_until_save(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_state.mufinder", "UEPState")
    window._document = doc

    job_id = UUID("11111111-1111-1111-1111-111111111111")
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")
    window._on_uep_analysis_state_changed(
        job_id,
        {
            "current_tab": 3,
            "probe_output": "saved",
            "clustering_window_state": {
                "is_open": True,
                "method": "kmeans",
                "kmeans_k": 2,
                "viewer_state": {"show_bonds": True},
            },
        },
    )

    stored = doc.ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert stored[str(job_id)]["current_tab"] == 3
    assert stored[str(job_id)]["probe_output"] == "saved"
    assert stored[str(job_id)]["clustering_window_state"]["viewer_state"]["show_bonds"] is True
    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text

    service.save_project(doc)

    reloaded = service.load_project(doc.project_root)
    persisted = reloaded.ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert persisted[str(job_id)]["current_tab"] == 3
    assert persisted[str(job_id)]["probe_output"] == "saved"
    assert persisted[str(job_id)]["clustering_window_state"]["method"] == "kmeans"
    window.close()


def test_main_window_uep_potential_grid_ready_updates_document_but_not_disk_until_save(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_grid_state.mufinder", "UEPGridState")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[run.id] = run

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(_fixture_text("Si2.castep"), encoding="utf-8")
    (output_dir / f"{job.label}.den_fmt").write_text(_fixture_text("Si2.den_fmt"), encoding="utf-8")
    (output_dir / f"{job.label}.cell").write_text(_fixture_text("Si2.cell"), encoding="utf-8")

    grid = potential_grid(
        UepField(
            load_uep_source(
                den_fmt_path=output_dir / f"{job.label}.den_fmt",
                castep_path=output_dir / f"{job.label}.castep",
                cell_path=output_dir / f"{job.label}.cell",
            )
        )
    )

    initial_results = sorted((doc.project_root / "entities" / "results").glob("*.json"))
    window._on_uep_potential_grid_ready(job.id, main_window_module.serialize_potential_grid(grid))

    persisted = window._uep_potential_grid_result_for_job(job.id)
    assert persisted is not None
    assert persisted.result_type == window._UEP_POTENTIAL_GRID_RESULT_TYPE
    assert window._load_uep_potential_grid_for_job(job.id) is not None
    assert sorted((doc.project_root / "entities" / "results").glob("*.json")) == initial_results

    service.save_project(doc)

    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_persisted = reloaded_window._uep_potential_grid_result_for_job(job.id)
    assert reloaded_persisted is not None
    assert reloaded_persisted.result_type == window._UEP_POTENTIAL_GRID_RESULT_TYPE
    reloaded_window.close()
    window.close()


def test_main_window_exposes_uep_cluster_rows_for_generation(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_cluster_generation.mufinder", "UEPClusterGeneration")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=doc.project_root / "files" / "uep",
    )
    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    run = CalculationRun(definition_id=definition.id, job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[run.id] = run
    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {
        str(job.id): {
            "batch_results": [
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
            "batch_summary_data": {
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
        }
    }
    window._document = doc

    rows = window._uep_cluster_source_rows()

    assert len(rows) == 1
    representative = rows[0]
    assert representative["generation_method"] == "uep_cluster"
    assert representative["selection_kind"] == "cluster"
    assert representative["source_structure_id"] == str(host.id)
    assert representative["properties"]["classification"] == "minimum"

    window._on_generate_requested("uep_clusters", {"selections": [representative]})

    assert len(doc.sites) == 1
    site = next(iter(doc.sites.values()))
    assert site.generation_method == "uep_cluster"
    assert site.source_structure_id == host.id
    assert site.properties["classification"] == "minimum"

    window.close()


def test_main_window_ignores_stale_uep_cluster_states_for_generation(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_stale_cluster_generation.mufinder", "UEPStaleClusterGeneration")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    stale_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
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
        parameters={"name_prefix": "uep", "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=doc.project_root / "files" / "uep",
    )
    active_job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    active_definition.parameters["job_ids_by_label"] = {"row_1": str(active_job.id)}
    active_run = CalculationRun(definition_id=active_definition.id, job_id=active_job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[active_run.id] = active_run
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(active_definition.id)

    def cluster_state(cluster_index: int, fractional_x: float) -> dict:
        return {
            "batch_results": [
                {
                    "optimized_position_ang": [fractional_x, 1.0, 1.5],
                    "optimized_fractional_position": [fractional_x, 0.2, 0.3],
                    "classical_energy_ev": -0.7,
                    "classification": {"label": "minimum"},
                }
            ],
            "batch_summary_data": {
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
        }

    doc.ui_state[window._UEP_ANALYSIS_STATES_KEY] = {
        str(stale_job.id): cluster_state(99, 0.9),
        str(active_job.id): cluster_state(1, 0.1),
    }
    window._document = doc

    rows = window._uep_cluster_source_rows()

    assert rows
    assert {row["source_job_id"] for row in rows} == {str(active_job.id)}
    assert {row["cluster_index"] for row in rows} == {1}

    window.close()


class _SignalStub:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


def _fixture_text(name: str) -> str:
    return (Path("tests/fixtures/uep") / name).read_text(encoding="utf-8")
