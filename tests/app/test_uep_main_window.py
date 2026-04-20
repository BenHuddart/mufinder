"""Focused app-level tests for the UEP workflow integration."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

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
            self.promote_requested = _SignalStub()
            self.state_changed = _SignalStub()
            self.potential_grid_ready = _SignalStub()

        def exec(self) -> int:
            captured["executed"] = True
            return 1

        def state_snapshot(self) -> dict[str, object]:
            return {"current_tab": 2}

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)
    window._on_open_uep_analysis_requested("row_1")

    assert captured["source_job_label"] == job.label
    assert captured["source_structure_id"] == host.id
    assert captured["source_structure"] is host
    assert captured["initial_state"] == {"current_tab": 2}
    assert captured["initial_potential_grid"] is None
    assert captured["executed"] is True
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
            self.promote_requested = _SignalStub()
            self.state_changed = _SignalStub()
            self.potential_grid_ready = _SignalStub()

        def exec(self) -> int:
            return 1

        def state_snapshot(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(main_window_module, "UepAnalysisDialog", _FakeDialog)
    window._on_open_uep_analysis_requested("row_1")

    assert captured["initial_potential_grid"] is not None
    window.close()


def test_main_window_uep_analysis_state_changed_persists_by_job_id(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_state.mufinder", "UEPState")
    window._document = doc

    job_id = UUID("11111111-1111-1111-1111-111111111111")
    window._on_uep_analysis_state_changed(job_id, {"current_tab": 3, "probe_output": "saved"})

    stored = doc.ui_state[window._UEP_ANALYSIS_STATES_KEY]
    assert stored[str(job_id)]["current_tab"] == 3
    assert stored[str(job_id)]["probe_output"] == "saved"
    window.close()


def test_main_window_uep_potential_grid_ready_persists_result(qapp, tmp_path: Path) -> None:
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

    window._on_uep_potential_grid_ready(job.id, main_window_module.serialize_potential_grid(grid))

    persisted = window._uep_potential_grid_result_for_job(job.id)
    assert persisted is not None
    assert persisted.result_type == window._UEP_POTENTIAL_GRID_RESULT_TYPE
    assert window._load_uep_potential_grid_for_job(job.id) is not None
    window.close()


def test_main_window_promote_uep_candidate_creates_site(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_promote.mufinder", "UEPPromote")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._on_promote_uep_candidate_requested(
        {
            "label_hint": "UEP Cluster 1",
            "fractional_position": [0.1, 0.2, 0.3],
            "cartesian_position": [0.5, 1.0, 1.5],
            "generation_method": "uep_batch_cluster",
            "source_structure_id": str(host.id),
            "source_job_label": "uep_row1",
            "properties": {"classification": "minimum", "total_energy_ev": -0.5},
        }
    )

    assert len(doc.sites) == 1
    site = next(iter(doc.sites.values()))
    assert site.generation_method == "uep_batch_cluster"
    assert site.source_structure_id == host.id
    assert site.properties["classification"] == "minimum"
    window.close()


class _SignalStub:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


def _fixture_text(name: str) -> str:
    return (Path("tests/fixtures/uep") / name).read_text(encoding="utf-8")
