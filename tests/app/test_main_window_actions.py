"""Behavior tests for main-window command flows and error handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write

import mufinder.app.main_window as main_window_module
import mufinder.services.job_artifacts as job_artifacts_module
from mufinder.app.workflow_controllers.base import output_predates_run
from mufinder.castep_io.output_parser import CastepResult
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, ComparisonTable, Job, MuonSite, Result, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.job_paths import definition_output_directory
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.qt import find_item_by_route
from tests.support.main_window import (
    select_route,
    stage_quantum_row_with_singlepoint,
)


def test_new_project_success(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()

    monkeypatch.setattr(main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("MyProj", True))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    window._new_project()

    assert window._document is not None
    assert window._document.project.name == "MyProj"
    assert (tmp_path / "MyProj.mufinder" / "project.json").exists()
    window.close()


def test_new_project_existing_non_empty_warns(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "Exists.mufinder"
    root.mkdir()
    (root / "dummy.txt").write_text("x", encoding="utf-8")

    window = MainWindow()
    calls: list[str] = []

    monkeypatch.setattr(main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("Exists", True))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: calls.append("warn"))

    window._new_project()

    assert calls == ["warn"]
    window.close()


def test_open_invalid_project_shows_error(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "not_project"
    folder.mkdir()

    window = MainWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(folder),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: errors.append("critical"))

    window._open_project()
    assert errors == ["critical"]
    window.close()


def test_main_window_recent_projects_tracks_latest_first(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, object] = {}

        def value(self, key: str, default=None, value_type=None):  # noqa: ANN001 - Qt compatibility.
            return self._values.get(key, default)

        def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001 - Qt API compatibility.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    service = ProjectService()
    first = service.create_project(tmp_path / "first_recent.mufinder", "FirstRecent")
    second = service.create_project(tmp_path / "second_recent.mufinder", "SecondRecent")

    window = MainWindow()

    assert window._open_project_from_path(first.project_root)
    assert window._open_project_from_path(second.project_root)

    recents = window._recent_projects()
    assert recents[:2] == [str(second.project_root), str(first.project_root)]

    menu_actions = [action.text() for action in window._recent_projects_menu.actions() if action.text()]
    assert str(second.project_root) in menu_actions
    assert str(first.project_root) in menu_actions
    assert "Clear Recent Projects" in menu_actions
    window.close()


def test_open_recent_project_removes_missing_entry(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, object] = {}

        def value(self, key: str, default=None, value_type=None):  # noqa: ANN001 - Qt compatibility.
            return self._values.get(key, default)

        def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001 - Qt API compatibility.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    window = MainWindow()
    missing_path = str(tmp_path / "missing_recent.mufinder")
    window._set_recent_projects([missing_path])
    window._refresh_recent_projects_menu()

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warn"))

    window._open_recent_project(missing_path)

    assert warnings == ["warn"]
    assert window._recent_projects() == []
    window.close()


def test_file_dialogs_use_scoped_last_selected_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, str] = {}

        def value(self, key: str, default: str = "", value_type: type[str] | None = None) -> str:
            value = self._values.get(key, default)
            return str(value) if value_type is str else value

        def setValue(self, key: str, value: str) -> None:  # noqa: N802 - Qt API naming.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    project_root = tmp_path / "dialog_reuse.mufinder"
    ProjectService().create_project(project_root, "DialogReuse")

    window = MainWindow()

    chosen_dirs: list[str] = []

    def fake_get_existing_directory(*args, **kwargs) -> str:
        start_dir = args[2] if len(args) > 2 else kwargs.get("dir", "")
        chosen_dirs.append(start_dir)
        if len(chosen_dirs) == 1:
            return str(project_root)
        return ""

    source_dir = tmp_path / "imports"
    source_dir.mkdir()
    source = source_dir / "host.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)

    structure_dialog_dirs: list[str] = []

    def fake_get_open_file_name(*args, **kwargs) -> tuple[str, str]:
        start_dir = args[2] if len(args) > 2 else kwargs.get("dir", "")
        structure_dialog_dirs.append(start_dir)
        return (str(source), "Structure Files")

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        fake_get_existing_directory,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    window._open_project()
    window._on_load_structure_requested()
    window._open_project()

    assert structure_dialog_dirs == [str(project_root)]
    assert len(chosen_dirs) == 2
    assert chosen_dirs[1] == str(project_root)
    window.close()


def test_save_without_document_shows_info(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    infos: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))

    window._save_project()

    assert infos == ["info"]
    window.close()


def test_generate_requested_handles_errors(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "gen_ui.mufinder", "GenUI")

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warning"))

    def boom(*args, **kwargs):
        raise SiteGenerationError("boom")

    monkeypatch.setattr(window._project_service, "generate_candidate_sites", boom)
    window._on_generate_requested("manual", {"coord_system": "fractional", "x": 0.0, "y": 0.0, "z": 0.0})

    assert warnings == ["warning"]
    window.close()


def test_delete_muon_candidates_requested_removes_selected_rows(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "delete_candidate_ui.mufinder", "DeleteCandidateUI")
    candidate = MuonSite(label="Candidate", status="generated", generation_method="manual")
    relaxed = MuonSite(label="Relaxed", status="relaxed", generation_method="manual")
    window._document.sites[candidate.id] = candidate
    window._document.sites[relaxed.id] = relaxed
    service.save_project(window._document)
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._on_delete_muon_candidates_requested([str(candidate.id), str(relaxed.id), "not-a-uuid"])

    assert save_calls == ["save"]
    assert candidate.id not in window._document.sites
    assert relaxed.id in window._document.sites
    reloaded = service.load_project(window._document.project_root)
    assert candidate.id not in reloaded.sites
    assert relaxed.id in reloaded.sites
    assert window._muon_generation_view._site_table.rowCount() == 0
    window.close()


def test_cluster_muon_positions_requested_updates_rows(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "cluster_candidate_ui.mufinder", "ClusterCandidateUI")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(window._document, host)
    site = MuonSite(
        label="Candidate",
        status="generated",
        generation_method="manual",
        fractional_position=[0.1, 0.2, 0.3],
        cartesian_position=[0.5, 1.0, 1.5],
    )
    window._document.sites[site.id] = site

    window._on_cluster_muon_positions_requested(
        {
            "site_ids": [str(site.id)],
            "anchor_fractional": [0.9, 0.8, 0.7],
        }
    )

    assert site.fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert window._muon_generation_view._site_table.item(0, 3).text() == "0.9000"
    assert window._muon_generation_view._selected_site_ids() == [str(site.id)]
    window.close()


def test_muon_generation_table_uses_natural_candidate_label_order(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "candidate_order_ui.mufinder", "CandidateOrderUI")
    for label in ("Random Site 1", "Random Site 10", "Random Site 2"):
        site = MuonSite(label=label, status="generated", generation_method="random")
        window._document.sites[site.id] = site

    window._refresh_muon_generation_table()

    table = window._muon_generation_view._site_table
    assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
        "Random Site 1",
        "Random Site 2",
        "Random Site 10",
    ]
    window.close()


def test_random_generation_runs_in_worker_and_updates_estimated_distance(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "async_random_ui.mufinder", "AsyncRandomUI")
    host = Structure(
        label="Host",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(window._document, host)
    window._muon_generation_view._random_by_count.setChecked(True)
    window._muon_generation_view._random_count.setValue(1)

    window._on_generate_requested(
        "random",
        {
            "selection_mode": "count",
            "count": 1,
            "min_muon_atom": 0.8,
            "max_attempts": 5000,
            "seed": "3",
            "include_symmetry_equivalents": False,
        },
    )

    assert window._muon_generation_thread is not None
    assert window._muon_generation_view._generation_progress.maximum() == 0

    for _ in range(2000):
        qapp.processEvents()
        if window._muon_generation_thread is None:
            break

    assert window._muon_generation_thread is None
    assert len(window._document.sites) == 1
    site = next(iter(window._document.sites.values()))
    assert site.properties["estimated_min_muon_muon"] > 0.0
    assert window._muon_generation_view._random_min_muon_muon.value() == pytest.approx(
        site.properties["estimated_min_muon_muon"]
    )
    assert window._muon_generation_view._generation_progress.value() == 1
    assert window._muon_generation_view._generation_progress.maximum() == 1
    assert window._muon_generation_view._generation_progress.format() == "Generated 1 site(s)"
    assert window._muon_generation_view._generate_btn.isEnabled()
    window.close()


def test_random_generation_progress_text_omits_predicted_total_for_separation_mode(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "random_progress_text.mufinder", "RandomProgressText")

    window._on_generation_progress_changed("random", 6, 13, "separation")

    assert window._muon_generation_view._generation_progress.value() == 6
    assert window._muon_generation_view._generation_progress.maximum() == 13
    assert window._muon_generation_view._generation_progress.format() == "Generated %v sites"

    window._on_generation_progress_changed("random", 6, 13, "count")

    assert window._muon_generation_view._generation_progress.format() == "Generated %v/%m sites"
    window.close()


def test_add_demo_job_requires_project(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    infos: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))
    window._add_demo_job()
    assert infos == ["info"]
    window.close()


def test_host_structure_route_switches_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "route_test.mufinder", "Route")
    window._refresh_ui_after_project_change()

    host_item = window._project_tree.topLevelItem(0)
    assert host_item is not None
    window._project_tree.setCurrentItem(host_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._host_structure_view
    window.close()


def test_load_structure_requested_imports_and_updates_view(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "load_test.mufinder", "Load")
    window._refresh_ui_after_project_change()

    source = tmp_path / "host.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(source), "Structure Files"),
    )

    select_route(window, "host_structure")
    assert window._main_stack.currentWidget() is window._host_structure_view

    window._on_load_structure_requested()

    assert window._document.project.host_structure_id is not None
    assert window._host_structure_view._atom_table.rowCount() == 1
    assert window._main_stack.currentWidget() is window._host_structure_view
    window.close()


def test_generate_comparison_uses_selected_job_labels_for_dft_columns(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "comparison_labels.mufinder", "ComparisonLabels")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    dft = Structure(
        label="Relaxed",
        lattice_vectors=[[5.1, 0.0, 0.0], [0.0, 5.1, 0.0], [0.0, 0.0, 5.1]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    window._project_service.add_structure(window._document, host)
    window._project_service.add_structure(window._document, dft)
    window._document.project.host_structure_id = host.id

    completed = CompletedPristineJob(
        job_id=uuid4(),
        job_label="pristine_relax_alpha",
        structure_id=dft.id,
        supercell_label="Unit cell",
        status="completed",
        total_energy=None,
        lattice_a=None,
        lattice_b=None,
        lattice_c=None,
        parameter_values={},
    )

    class FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def selected_job_ids(self) -> list[UUID]:
            return [completed.job_id]

        def table_label(self) -> str:
            return "Comparison 1"

    monkeypatch.setattr(window, "_completed_pristine_relaxation_jobs", lambda: [completed])
    monkeypatch.setattr(main_window_module, "ComparisonJobDialog", FakeDialog)
    monkeypatch.setattr(window, "_sync_comparison_table_views", lambda: None)
    monkeypatch.setattr(window, "_populate_project_tree", lambda: None)
    monkeypatch.setattr(window, "_set_active_route", lambda route: None)
    monkeypatch.setattr(
        main_window_module,
        "compare_structures",
        lambda *_args, **_kwargs: SimpleNamespace(
            definitions=(),
            experimental_values={},
            dft_columns=(),
            warnings=(),
        ),
    )

    window._on_generate_comparison_table_requested()

    tables = list(window._document.comparison_tables.values())
    assert len(tables) == 1
    assert tables[0].dft_labels == ["DFT (pristine_relax_alpha)"]

    window.close()


def test_comparison_table_title_edit_updates_project_tree_and_persists(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "comparison_title_edit.mufinder", "ComparisonTitleEdit")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    table = ComparisonTable(
        label="Original title",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "metric_definitions": [{"key": "a", "label": "a", "unit": "Ang", "group": "Lattice"}],
            "experimental_values": {"a": {"display": "5"}},
            "dft_columns": [{"label": "DFT-1", "values": {"a": {"display": "5.1"}}}],
        },
    )
    service.add_comparison_table(doc, table)
    service.save_project(doc)
    window._document = doc
    window._sync_comparison_table_views()
    window._populate_project_tree()
    window._set_active_route(window._comparison_route(table.id))

    widget = window._comparison_widgets[table.id]
    widget._title_edit.setText("  Renamed title  ")
    widget._title_edit.editingFinished.emit()

    assert doc.comparison_tables[table.id].label == "Renamed title"
    tree_item = window._find_tree_item_by_route(window._comparison_route(table.id))
    assert tree_item is not None
    assert tree_item.text(0) == "Renamed title"
    reloaded = service.load_project(doc.project_root)
    assert reloaded.comparison_tables[table.id].label == "Renamed title"
    window.close()


def test_host_magnetic_configuration_changes_update_document(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "host_spin_update.mufinder", "HostSpin")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={},
    )
    window._project_service.add_structure(window._document, host)
    window._document.project.host_structure_id = host.id

    window._on_host_magnetic_configuration_changed(
        {
            "atomic_spins": [2.0],
            "propagation_vector": [0.0, 0.0, 0.25],
            "magnetic_structure": {
                "propagation_vector": [0.0, 0.0, 0.25],
                "moments": [{"label": "Fe1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"}],
            },
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        }
    )

    assert host.properties["atomic_spins"] == pytest.approx([2.0])
    assert host.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert window._document.project.magnetic_structure is not None
    assert window._document.project.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    window._project_service.save_project(window._document)
    reloaded = window._project_service.load_project(window._document.project_root)
    reloaded_host = reloaded.structures[host.id]
    assert reloaded_host.properties["atomic_spins"] == pytest.approx([2.0])
    assert reloaded_host.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert reloaded.project.magnetic_structure is not None
    assert reloaded.project.magnetic_structure["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert reloaded.project.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    window.close()


def test_load_structure_requested_handles_import_error(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "load_error_test.mufinder", "LoadErr")

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("bad.cif", "Structure Files"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warn"))

    def broken_import(*args, **kwargs):
        raise StructureLoadError("bad")

    monkeypatch.setattr(window._project_service, "import_structure_file", broken_import)

    window._on_load_structure_requested()
    assert warnings == ["warn"]
    window.close()


def test_convergence_configured_registers_snapshot(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_refresh_convergence_simulation_cell_without_document_disables_selector(qapp) -> None:
    window = MainWindow()

    window._refresh_convergence_simulation_cell()

    assert not window._convergence_view._simulation_cell_combo.isEnabled()
    assert "Load a host structure" in window._convergence_view._simulation_cell_summary.text()
    window.close()


def test_convergence_simulation_cell_create_requested_handles_missing_context(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_convergence_simulation_cell_use_unit_cell_requested_resets_selection(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_convergence_supercell_initial_matrix_prefers_selected_asset(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_supercell_editor_route_uses_project_page(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "supercell_route.mufinder", "SupercellRoute")
    window._document = doc
    window._populate_project_tree()

    window._set_active_route("supercell_editor")

    assert window._main_stack.currentWidget() == window._supercell_editor_page
    window.close()


def test_convergence_simulation_cell_selection_clones_other_workflow_asset(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_convergence_configured_uses_selected_saved_supercell(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_convergence_configured_reuses_matching_definition(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_materialize_convergence_jobs_writes_saved_supercell_structure(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_main_window_syncs_convergence_row_status_and_energy(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_main_window_syncs_convergence_row_max_force(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_main_window_deserializes_convergence_config_with_structured_values(qapp) -> None:
    window = MainWindow()

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


def test_main_window_deserializes_cutoff_config_prefers_cutoff_energies_when_fields_diverge(qapp) -> None:
    window = MainWindow()

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


def test_main_window_convergence_definition_helpers_support_legacy_labels(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_main_window_detach_convergence_job_clears_job_id_mapping(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_open_castep_setup_dialog_updates_status_bar(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()

    class FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs) -> None:
            self._setup = type(
                "Setup",
                (),
                {
                    "executable_path": "/opt/castep/bin/castep",
                    "keywords_json_path": "/tmp/castep_keywords.json",
                    "mpirun_path": "",
                },
            )()

        def exec(self):
            return 1

        def setup(self):
            return self._setup

    monkeypatch.setattr(main_window_module, "CastepSetupDialog", FakeDialog)

    window._open_castep_setup_dialog()

    assert "CASTEP setup updated" in window.statusBar().currentMessage()
    window.close()


def test_journal_write_does_not_change_active_workflow_view_on_submission(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "journal_route_preserved.mufinder", "JournalRoutePreserved")
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
    output_dir = doc.project_root / "files" / "convergence_cutoff"
    (output_dir / job.label).mkdir(parents=True, exist_ok=True)
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_dir,
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: None)

    select_route(window, "host_structure")
    assert window._main_stack.currentWidget() is window._host_structure_view

    window._submit_jobs_locally([job])

    assert window._main_stack.currentWidget() is window._host_structure_view
    assert window._active_route() == "host_structure"
    window.close()


def test_main_window_project_tree_uses_journal_route_and_view(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    doc = ProjectService().create_project(tmp_path / "journal_route.mufinder", "JournalRoute")
    window._document = doc
    window._refresh_ui_after_project_change()

    journal_item = None
    for index in range(window._project_tree.topLevelItemCount()):
        item = window._project_tree.topLevelItem(index)
        if item is not None:
            journal_item = find_item_by_route(item, "provenance_log")
        if journal_item is not None:
            break
    assert journal_item is not None
    assert journal_item.text(0) == "Journal"

    select_route(window, "provenance_log")

    assert window._main_stack.currentWidget() is window._journal_view
    window.close()


def test_main_window_journal_populates_from_imported_provenance(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "journal_populated.mufinder", "JournalPopulated")
    window._document = doc

    source = tmp_path / "host.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)
    service.import_structure_file(doc, source, make_active=True)

    window._refresh_ui_after_project_change()
    select_route(window, "provenance_log")

    texts = [label.text() for label in window._journal_view.findChildren(main_window_module.QLabel)]
    expected_day = doc.provenance[-1].timestamp.astimezone().strftime("%A %d %B %Y")
    assert expected_day in texts
    assert "Host structure imported" in texts
    window.close()


def test_main_window_local_finish_pristine_relaxation_uses_first_three_lattice_parameters(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_finish_pristine_relax.mufinder", "LocalFinishPristineRelax")

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
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}-out.cell").write_text("dummy", encoding="utf-8")

    # The relaxed -out.cell is parsed once by the off-thread collector
    # (mufinder.services.job_artifacts), so patch the collector's parse helpers
    # rather than main_window's (which no longer parses secondary outputs).
    monkeypatch.setattr(
        job_artifacts_module,
        "parse_cell_file",
        lambda _path: SimpleNamespace(
            lattice_vectors=((5.1, 0.0, 0.0), (0.0, 5.2, 0.0), (0.0, 0.0, 5.3)),
            species=("Si",),
            fractional_positions=((0.0, 0.0, 0.0),),
        ),
    )
    monkeypatch.setattr(
        job_artifacts_module,
        "lattice_parameters",
        lambda _vectors: (5.1, 5.2, 5.3, 90.0, 90.0, 90.0),
    )

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert run.status == JobStatus.COMPLETED
    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["lattice_a"] == pytest.approx(5.1)
    assert result.data["lattice_b"] == pytest.approx(5.2)
    assert result.data["lattice_c"] == pytest.approx(5.3)
    assert len(result.derived_structure_ids) == 1
    window.close()


def test_main_window_local_finish_pristine_relaxation_missing_out_cell_fails_job(
    qapp, tmp_path: Path
) -> None:
    """PristineRelaxationController.validate_artifacts, dispatched from
    _on_local_job_finished, must fail a pristine job whose relaxed structure
    output is absent BEFORE build_result is ever reached."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_missing_out.mufinder", "PristineMissingOut")

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
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    # Output directory exists but the relaxed -out.cell was NOT written.
    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,  # clears the geom prologue so we reach the out.cell guard
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.FAILED
    assert run.result_id is None
    assert not any(
        structure.origin == "pristine_relaxation" for structure in doc.structures.values()
    )
    window.close()


_PRISTINE_OUT_CELL_FIRST = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


_PRISTINE_OUT_CELL_CONTINUED = """
%BLOCK LATTICE_CART
ang
5.4 0.0 0.0
0.0 5.5 0.0
0.0 0.0 5.6
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.01 0.02 0.03
%ENDBLOCK POSITIONS_FRAC
"""


def _pristine_relaxation_job_fixture(window: MainWindow, service: ProjectService, tmp_path: Path, name: str):
    """Build a document with one running pristine-relaxation job and its output dir."""

    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
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
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    window._document = doc
    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    return doc, host, job, output_dir


def _start_pristine_run(window: MainWindow, service: ProjectService, doc, job, run_index: int):
    """Take ``job`` through submit → start and register a fresh active run."""

    service.submit_job(job)
    service.start_job(job)
    run = CalculationRun(job_id=job.id, run_index=run_index, status=JobStatus.RUNNING)
    run.started_at = job.started_at
    doc.runs[run.id] = run
    window._active_run_by_job[job.id] = run.id
    return run


def test_main_window_local_finish_relaxation_stale_out_cell_fails_job(
    qapp, tmp_path: Path
) -> None:
    """A continued run that dies before writing a new ``-out.cell`` leaves the
    PREVIOUS run's file in the re-used job directory, where the completion path
    would happily validate it as this run's geometry. The relaxation
    controller's stale-output check fails the job instead, so no result is built from a superseded file."""

    window = MainWindow()
    service = ProjectService()
    doc, _host, job, output_dir = _pristine_relaxation_job_fixture(
        window, service, tmp_path, "StaleOutCell"
    )

    # The previous run's relaxed structure, left behind in the re-used directory.
    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_FIRST, encoding="utf-8"
    )
    run = _start_pristine_run(window, service, doc, job, run_index=2)
    # This run started well after that file was written.
    run.started_at = datetime.now(UTC) + timedelta(minutes=5)

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,  # clears the geom prologue so we reach the stale guard
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.FAILED
    assert "predates this run" in (job.error_message or "")
    assert run.result_id is None
    assert not any(
        structure.origin == "pristine_relaxation" for structure in doc.structures.values()
    )
    window.close()


def test_main_window_stale_output_guard_skips_jobs_with_no_start_time(tmp_path: Path) -> None:
    """The guard fails open. A job the poller never observed running has no
    ``started_at`` to compare against, so its output is accepted rather than a
    job being failed on no evidence. Sub-slack differences are also accepted, so
    filesystem mtime granularity cannot false-fail a fresh output."""

    path = tmp_path / "seed-out.cell"
    path.write_text("cell", encoding="utf-8")
    written = datetime.fromtimestamp(path.stat().st_mtime, UTC)

    assert output_predates_run(path, None) is False
    assert output_predates_run(None, written) is False
    # Written after the run started, and written a hair before it: both fresh.
    assert output_predates_run(path, written - timedelta(minutes=5)) is False
    assert output_predates_run(path, written + timedelta(seconds=1)) is False
    # A whole run older: stale.
    assert output_predates_run(path, written + timedelta(minutes=5)) is True
    # An unreadable path is not evidence of staleness either.
    assert output_predates_run(tmp_path / "absent", written) is False


def test_main_window_pristine_relaxation_recompletion_keeps_trail_and_references(
    qapp, tmp_path: Path
) -> None:
    """Drive a full continue → running → re-complete cycle through the real
    completion path.

    The pristine-relaxation workflow-trail milestone is table-driven
    (``_workflow_pristine_relaxation_complete`` reads ``document.comparison_tables``),
    **not** job-driven, so it stays complete across the whole cycle — continuing a
    job neither un-completes nor re-completes it. What the cycle really has to
    protect is the reference that milestone stands for: the comparison table's
    ``dft_structure_ids`` must still resolve, which holds only because the second
    completion updates the derived structure in place instead of minting a new one.
    """

    window = MainWindow()
    service = ProjectService()
    doc, host, job, output_dir = _pristine_relaxation_job_fixture(
        window, service, tmp_path, "PristineRecompletion"
    )
    parsed = CastepResult(
        completed=True, total_energy=-11.0, warnings=(), geom_converged=True
    )

    run_one = _start_pristine_run(window, service, doc, job, run_index=1)
    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_FIRST, encoding="utf-8"
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    first_result = doc.results[run_one.result_id]
    derived_id = first_result.derived_structure_ids[0]
    structure_count = len(doc.structures)

    # A comparison table built from that first completion.
    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[derived_id],
        dft_labels=[job.label],
    )
    service.add_comparison_table(doc, table)
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    # Continue the completed job and start its second run.
    service.prepare_job_continuation(job)
    assert job.status == JobStatus.DEFINED
    run_two = _start_pristine_run(window, service, doc, job, run_index=2)
    assert job.status == JobStatus.RUNNING
    # Table-driven, so the milestone does not dip while the continued job runs.
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_CONTINUED, encoding="utf-8"
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    # One derived structure, same id, refreshed geometry; both Results agree.
    second_result = doc.results[run_two.result_id]
    assert second_result.id != first_result.id
    assert second_result.derived_structure_ids == [derived_id]
    assert len(doc.structures) == structure_count
    assert second_result.data["lattice_a"] == pytest.approx(5.4)
    derived = doc.structures[derived_id]
    assert derived.lattice_vectors == [[5.4, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 5.6]]

    # The comparison table's reference survived the cycle intact.
    assert table.dft_structure_ids == [derived_id]
    assert service.validate_document(doc) == []
    window.close()


def test_main_window_muon_job_uses_row_label_and_charge_seed_directory(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
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


def test_materialize_muon_relaxation_filters_charge_qualified_selection(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_muon_relaxation_staging_updates_memory_without_autosave(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_muon_relaxation_staging_rejected_charge_dialog_leaves_site_generated(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_muon_relaxation_staging_can_create_both_charge_state_tabs(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_muon_charge_state_dialog_defaults_and_returns_both_states(qapp) -> None:
    dialog = main_window_module.MuonChargeStateDialog(site_count=2)

    assert dialog.objectName() == "muonChargeStateDialog"
    assert dialog.windowTitle() == "Create Muon Relaxation Jobs"
    assert dialog.selected_charge_states() == ("positive_muon",)

    dialog._both.setChecked(True)

    assert dialog.selected_charge_states() == ("positive_muon", "muonium")


def test_muon_relaxation_configuration_updates_memory_without_autosave(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_muon_relaxation_form_state_change_remains_manual_save_state(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_main_window_jobs_table_refresh_syncs_muon_relaxation_status(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
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


def test_convergence_configured_without_project_shows_info(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_main_window_generate_requested_without_project_shows_info(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._on_generate_requested("random", {"count": 1})

    assert infos == ["No project"]
    window.close()


def test_main_window_reset_quantum_corrections_to_stage_one_clears_shared_job_state(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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

    definition = window._ensure_quantum_corrections_definition([str(site.id)])
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

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda self: main_window_module.QMessageBox.StandardButton.Yes,
    )

    window._on_reset_quantum_corrections_to_stage_one([str(site.id)])

    assert singlepoint_job.id not in doc.jobs
    assert definition.parameters["singlepoint_job_ids_by_label"] == {}
    assert definition.parameters["job_seed_names_by_label"] == {}
    assert row_settings[str(site.id)]["num_farms"] == 4
    assert output_dir.exists() is False

    candidate = window._quantum_source_candidate_by_key(str(site.id))
    assert candidate is not None
    stage_state = window._quantum_stage_state(definition, candidate)
    assert stage_state.stage == "singlepoint"
    assert stage_state.status == "ready"

    window.close()


def test_main_window_convergence_failed_handlers_forward_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_main_window_close_prompts_to_save_visible_project(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "close_prompt.mufinder", "ClosePrompt")

    saves: list[bool] = []
    shutdowns: list[bool] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: shutdowns.append(True))

    window.show()
    window.close()

    assert saves == [True]
    assert shutdowns == [True]


def test_main_window_close_cancel_keeps_window_open(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    window._document = service.create_project(tmp_path / "close_cancel.mufinder", "CloseCancel")

    shutdowns: list[bool] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Cancel,
    )
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: shutdowns.append(True))

    window.show()
    window.close()

    assert shutdowns == []
    assert window.isVisible() is True
    window._document = None
    window.close()


def test_main_window_submit_from_convergence_shows_info_when_no_matching_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_submit_none.mufinder", "ConvSubmitNone")
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._convergence_controller.submit([300.0], target="local")

    assert infos == ["No ready jobs"]
    window.close()


def test_main_window_configured_cutoff_shows_ready_and_stays_off_dashboard(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_main_window_submit_from_convergence_materializes_job_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_materialize.mufinder", "ConvMaterialize")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
    }
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def config(self):
            return SimpleNamespace(parallel=False, max_concurrent=1, num_cores=1)

    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: submitted.append(kwargs))

    window._convergence_controller.submit([300.0], target="local")

    assert len(doc.jobs) == 1
    job = next(iter(doc.jobs.values()))
    assert job.status == JobStatus.SUBMITTING
    assert submitted and submitted[0]["seed_name"] == "conv_ecut_300"
    assert (doc.project_root / "files" / "convergence_cutoff" / "conv_ecut_300" / "conv_ecut_300.cell").exists()
    assert (doc.project_root / "files" / "convergence_cutoff" / "conv_ecut_300" / "conv_ecut_300.param").exists()
    definition = next(iter(doc.calculations.values()))
    assert definition.parameters["job_ids_by_cutoff"][main_window_module.MainWindow._cutoff_key(300.0)] == str(job.id)
    window.close()


def test_main_window_submit_from_convergence_dedupes_existing_ready_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_submit_dedupe.mufinder", "ConvSubmitDedupe")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job_a = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_b = service.add_job(
        doc,
        label="conv_ecut_360",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_a.status = JobStatus.DEFINED
    job_b.status = JobStatus.DEFINED

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
        job_ids_by_cutoff={
            main_window_module.MainWindow._cutoff_key(300.0): str(job_a.id),
            main_window_module.MainWindow._cutoff_key(360.0): str(job_b.id),
        },
    )
    window._document = doc

    submitted_ids: list[list[object]] = []
    monkeypatch.setattr(
        window,
        "_submit_jobs_locally",
        lambda jobs: submitted_ids.append([job.id for job in jobs]),
    )

    window._convergence_controller.submit([300.0, 360.0], target="local")

    assert submitted_ids == [[job_a.id, job_b.id]]
    window.close()


def test_main_window_reset_failed_convergence_job_restores_ready_state(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    removed: list[list[float]] = []
    monkeypatch.setattr(window._convergence_view, "remove_cutoffs", lambda cutoffs: removed.append(list(cutoffs)))

    window._on_remove_cutoffs_from_convergence([300.0])

    assert removed == [[300.0]]
    window.close()


def test_main_window_remove_single_convergence_cutoff_warns_on_file_cleanup_error(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_main_window_remove_cutoff_from_definition_updates_lists_and_labels(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_main_window_file_helpers_cover_file_prompt_and_unique_archive(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMessageBox:
        Icon = SimpleNamespace(Warning=1)
        StandardButton = SimpleNamespace(Yes=1, No=2, Cancel=3, YesToAll=4, NoToAll=5)
        _answer = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def setIcon(self, *_args, **_kwargs) -> None:
            pass

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            pass

        def setText(self, *_args, **_kwargs) -> None:
            pass

        def setInformativeText(self, *_args, **_kwargs) -> None:
            pass

        def setStandardButtons(self, *_args, **_kwargs) -> None:
            pass

        def setDefaultButton(self, *_args, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return _FakeMessageBox._answer

    window = MainWindow()
    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") == "delete"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.No
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") == "archive"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Cancel
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") is None
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "delete"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.YesToAll
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "delete_all"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.NoToAll
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "archive_all"

    file_path = tmp_path / "orphan.txt"
    file_path.write_text("x", encoding="utf-8")
    window._delete_path(file_path)
    assert not file_path.exists()

    output_root = tmp_path / "files"
    source = output_root / "job_a"
    archive_root = output_root / "removed_from_workflow"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "job_a").mkdir()
    source.mkdir(parents=True, exist_ok=True)

    archived = window._archive_cutoff_directory(source, output_root)

    assert archived.name == "job_a_1"
    assert archived.exists()
    assert not source.exists()
    window.close()


def test_main_window_examine_convergence_launches_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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


def test_main_window_on_cutoff_adopted_sets_project_default(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_main_window_kpoint_examiner_state_changed_updates_ui_state_without_saving_project(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_examiner_state_update.mufinder", "KPointExaminerStateUpdate")
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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

    assert loaded.ui_state[window._KPOINT_EXAMINER_STATE_KEY]["results_text"] == "Recommended grid: 2 x 2 x 2"
    assert loaded.ui_state[window._KPOINT_EXAMINER_STATE_KEY]["plot_data"]["cutoffs"] == [8.0, 27.0]
    window.close()


def test_kpoint_convergence_configured_registers_snapshot(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_refresh_kpoint_simulation_cell_without_document_disables_selector(qapp) -> None:
    window = MainWindow()

    window._refresh_kpoint_convergence_simulation_cell()

    assert not window._kpoint_convergence_view._simulation_cell_combo.isEnabled()
    assert "Load a host structure" in window._kpoint_convergence_view._simulation_cell_summary.text()
    window.close()


def test_main_window_submit_from_kpoint_shows_info_when_no_matching_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_submit_none.mufinder", "KPointSubmitNone")
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._kpoint_convergence_controller.submit([(2, 2, 2)], target="local")

    assert infos == ["No ready jobs"]
    window.close()


def test_main_window_form_kpoint_shows_ready_before_submission(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_form_ready.mufinder", "KPointFormReady")
    doc.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2], [3, 3, 2]],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert window._kpoint_convergence_view._table.item(0, window._kpoint_convergence_view._STATUS_COLUMN).text() == "ready"
    assert window._kpoint_convergence_view._table.item(1, window._kpoint_convergence_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_submit_from_kpoint_materializes_job_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_materialize.mufinder", "KPointMaterialize")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def config(self):
            return SimpleNamespace(parallel=False, max_concurrent=1, num_cores=1)

    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: submitted.append(kwargs))

    window._kpoint_convergence_controller.submit([(2, 2, 2)], target="local")

    assert len(doc.jobs) == 1
    job = next(iter(doc.jobs.values()))
    assert job.status == JobStatus.SUBMITTING
    assert submitted and submitted[0]["seed_name"] == "kconv_kgrid_2x2x2"
    assert (doc.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.cell").exists()
    assert (doc.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param").exists()
    definition = next(iter(doc.calculations.values()))
    assert definition.parameters["job_ids_by_grid"][main_window_module.MainWindow._grid_key((2, 2, 2))] == str(job.id)
    window.close()


def test_materialize_pristine_relaxation_rewrites_defined_job_inputs_when_project_defaults_change(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_rewrite.mufinder", "PristineRewrite")

    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._on_pristine_relaxation_configured(
        main_window_module.PristineRelaxationConfig(
            name_prefix="pristine_relax",
            rows=(main_window_module.PristineRelaxationRow(label="row_1", parameter_values={}),),
        )
    )

    jobs = window._pristine_relaxation_controller.materialize(["row_1"])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == JobStatus.DEFINED

    cell_path = doc.project_root / "files" / "pristine_relaxation" / job.label / f"{job.label}.cell"
    initial_cell = cell_path.read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID" not in initial_cell

    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    jobs = window._pristine_relaxation_controller.materialize(["row_1"])
    assert len(jobs) == 1
    assert jobs[0].id == job.id

    refreshed_cell = cell_path.read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 4 4 4" in refreshed_cell
    window.close()


def test_on_pristine_relaxation_configured_surfaces_kpoint_message(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A supercell row rescales the project k-point grid; the adjustment message
    # from AdapterExpansion.messages must be surfaced to the user, not dropped.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_msg.mufinder", "PristineMsg")

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
    window._document = doc

    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, text, *args, **kwargs: infos.append((title, text)),
    )

    window._on_pristine_relaxation_configured(
        main_window_module.PristineRelaxationConfig(
            name_prefix="pristine_relax",
            rows=(
                main_window_module.PristineRelaxationRow(
                    label="super",
                    fix_all_cell=False,
                    supercell_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                    parameter_values={},
                ),
            ),
        )
    )

    assert any("K-point grid adjusted" in text for _title, text in infos)
    window.close()


def test_main_window_form_pristine_relaxation_shows_ready_before_submission(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_ready.mufinder", "PristineFormReady")

    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []},
            {"label": "row_2", "fix_all_cell": True, "params": {}, "added_advanced": []},
        ],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    assert window._pristine_relaxation_view._table.item(1, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_form_muon_relaxation_shows_ready_before_submission(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
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
    doc.ui_state["muon_relaxation_form"] = window._serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(row,))
    )

    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "defined"
    assert window._muon_relaxation_controller.build_states()["positivemuon:Site 1"].status == "defined"
    window.close()


def test_main_window_muon_relaxation_sync_prefers_defined_job_over_cancelled_history(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    serialized = window._serialize_muon_relaxation_config(config)
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    serialized = window._serialize_muon_relaxation_config(config)
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
    serialized = window._serialize_muon_relaxation_config(
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(
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
    doc.ui_state["muon_relaxation_form"] = window._serialize_muon_relaxation_config(
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


def test_main_window_pristine_form_edit_hides_stale_definition_status(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_stale.mufinder", "PristineFormStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": True, "params": {}, "added_advanced": []},
        ],
    }

    window._refresh_ui_after_project_change()

    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_pristine_form_edit_preserves_matching_row_statuses(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_partial_status.mufinder", "PristineFormPartialStatus")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    spinpol_row = main_window_module.PristineRelaxationRow(
        label="spinpol",
        parameter_values={"spin_polarized": True},
    )
    nospin_row = main_window_module.PristineRelaxationRow(label="nospin", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(
                name_prefix="pristine_relax",
                rows=(spinpol_row, nospin_row),
            )
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    completed_job = service.add_job(
        doc,
        label="pristine_relax_spinpol",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    completed_job.status = JobStatus.COMPLETED
    reset_job = service.add_job(
        doc,
        label="pristine_relax_nospin",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    reset_job.status = JobStatus.DEFINED
    definition.parameters["job_ids_by_label"] = {
        "spinpol": str(completed_job.id),
        "nospin": str(reset_job.id),
    }
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {
                "label": "spinpol",
                "fix_all_cell": False,
                "params": {"spin_polarized": True},
                "added_advanced": [],
            },
            {
                "label": "nospin",
                "fix_all_cell": False,
                "params": {"nextra_bands": 17},
                "added_advanced": ["nextra_bands"],
            },
        ],
    }

    window._refresh_ui_after_project_change()

    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "completed"
    assert window._pristine_relaxation_view._table.item(1, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_materialize_uep_rewrites_defined_job_inputs_when_project_defaults_change(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_pristine_relaxation_preserves_running_row_when_adding_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_running.mufinder", "PristineRunning")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row_1 = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    row_2 = main_window_module.PristineRelaxationRow(label="row_2", parameter_values={})
    old_config = main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row_1,))
    old_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(old_config),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    new_config = main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row_1, row_2))
    new_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(new_config),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    submitted: list[Job] = []
    monkeypatch.setattr(window, "_submit_jobs_on_cluster", lambda jobs: submitted.extend(jobs))

    window._pristine_relaxation_controller.submit(["row_2"], target="cluster")

    states = window._pristine_relaxation_controller.build_states()
    assert states["row_1"].status == JobStatus.RUNNING.value
    assert states["row_1"].job_id == running_job.id
    assert len(submitted) == 1
    assert submitted[0].label == "pristine_relax_row_2"
    assert new_definition.parameters["job_ids_by_label"]["row_1"] == str(running_job.id)
    window.close()


def test_muon_relaxation_preserves_running_row_across_reconfiguration(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._muon_relaxation_controller.build_states()

    assert states["positivemuon:Site 1"].status == JobStatus.RUNNING.value
    assert states["positivemuon:Site 1"].job_id == running_job.id
    assert new_definition.parameters["job_ids_by_label"][key] == str(running_job.id)
    assert new_definition.parameters["job_seed_names_by_label"][key] == "positivemuon_Site_1"
    window.close()


def test_uep_preserves_running_row_across_reconfiguration(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_pristine_relaxation_row_parameter_edit_keeps_row_bound_to_its_job(
    qapp, tmp_path: Path
) -> None:
    """In-place parameter edits must not sever the row→job binding.

    Registering a new definition would give the row an empty
    ``job_ids_by_label``; and because pristine relaxation has no active-status
    escape hatch in ``build_states``, a form/definition signature mismatch would
    drop the row to a jobless "ready". This drives the real widget, so the
    table's own parameter normalization (implicit project defaults are stripped)
    is exercised end to end.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_param_edit.mufinder", "PristineParamEdit")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(
        label="row_1", parameter_values={"geom_max_iter": 120}
    )
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    window._restore_pristine_relaxation_form_from_definition()
    assert window._pristine_relaxation_controller.build_states()["row_1"].job_id == job.id

    updated = window._pristine_relaxation_controller.update_row_parameters(
        "row_1", {"geom_max_iter": 500, "task": "GEOMETRYOPTIMIZATION"}
    )

    assert updated is True
    # No new definition; the binding is intact and the row still resolves its job.
    assert [
        candidate.id
        for candidate in doc.calculations.values()
        if candidate.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
    ] == [definition.id]
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    states = window._pristine_relaxation_controller.build_states()
    assert states["row_1"].job_id == job.id
    assert states["row_1"].status == JobStatus.COMPLETED.value

    config = window._deserialize_pristine_relaxation_config(
        definition.parameters["configuration"]
    )
    assert config.rows[0].parameter_values["geom_max_iter"] == 500
    # The widget is the normalizer of record: it strips the implicit ``task``
    # default, so the definition must not keep it either.
    assert "task" not in config.rows[0].parameter_values
    form_config = window._pristine_relaxation_form_config()
    assert window._pristine_relaxation_row_signature(
        form_config.rows[0]
    ) == window._pristine_relaxation_row_signature(config.rows[0])
    assert (
        doc.ui_state["pristine_relaxation_form"]["rows"][0]["params"]["geom_max_iter"]
        == 500
    )
    window.close()


def test_muon_relaxation_row_parameter_edit_touches_only_its_charge_state(
    qapp, tmp_path: Path
) -> None:
    """Two rows can share a label, so the edit is keyed by the charge-qualified
    row key and must leave the other charge state's row alone."""

    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(
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

    config = window._deserialize_muon_relaxation_config(
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


def test_pristine_relaxation_deleted_row_does_not_reinherit_stale_status(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_deleted_stale.mufinder", "PristineDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    old_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    window._on_remove_pristine_relaxation_rows(["row_1"])

    new_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._pristine_relaxation_controller.build_states()

    assert states["row_1"].status == "ready"
    assert new_definition.parameters.get("job_ids_by_label", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()


def test_pristine_relaxation_remove_row_deletes_associated_files_and_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_remove_files.mufinder", "PristineRemoveFiles")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    job_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    monkeypatch.setattr(window, "_prompt_job_file_disposition", lambda **_kwargs: "delete")

    window._on_remove_pristine_relaxation_rows(["row_1"])

    assert not job_dir.exists()
    assert job.id not in doc.jobs
    assert definition.id not in doc.calculations
    window.close()


def test_pristine_relaxation_remove_row_cancel_keeps_files_and_restores_form(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_remove_cancel.mufinder", "PristineRemoveCancel")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    job_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    window._pristine_relaxation_view.set_form_state({"name_prefix": "pristine_relax", "rows": []})
    monkeypatch.setattr(window, "_prompt_job_file_disposition", lambda **_kwargs: None)

    window._on_remove_pristine_relaxation_rows(["row_1"])

    assert job_dir.exists()
    assert job.id in doc.jobs
    assert definition.id in doc.calculations
    assert window._pristine_relaxation_view.collect_state()["rows"][0]["label"] == "row_1"
    window.close()


def test_muon_relaxation_deleted_row_does_not_reinherit_stale_status(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._muon_relaxation_controller.build_states()

    assert states["positivemuon:Site 1"].status == "ready"
    assert new_definition.parameters.get("job_ids_by_label", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()


def test_muon_relaxation_remove_row_archives_associated_files_and_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
        parameters=window._serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
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


def test_uep_deleted_row_does_not_reinherit_stale_status(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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


def test_kpoint_convergence_preserves_running_grid_across_reconfiguration(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


def test_kpoint_deleted_grid_does_not_reinherit_stale_status(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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

    label_a = main_window_module.convergence_job_name("kconv", (2, 2, 2), sweep_parameter="kpoint_mp_grid")
    label_b = main_window_module.convergence_job_name("kconv", (3, 3, 2), sweep_parameter="kpoint_mp_grid")
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
    monkeypatch.setattr(window, "_submit_jobs_locally", lambda jobs: submitted_ids.append([job.id for job in jobs]))

    window._kpoint_convergence_controller.submit([(2, 2, 2), (2, 2, 2), (3, 3, 2)], target="local")

    assert submitted_ids == [[job_a.id, job_b.id]]
    window.close()


def test_main_window_remove_kpoint_grids_without_document_delegates_to_widget(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    removed: list[list[tuple[int, int, int]]] = []
    monkeypatch.setattr(window._kpoint_convergence_view, "remove_grids", lambda grids: removed.append(list(grids)))

    window._on_remove_grids_from_kpoint_convergence([(2, 2, 2)])

    assert removed == [[(2, 2, 2)]]
    window.close()


def test_main_window_remove_kpoint_grids_updates_definition_configuration(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    assert window._kpoint_convergence_view._table.item(0, window._kpoint_convergence_view._KEY_COLUMN).text() == "3 x 3 x 2"
    window.close()


def test_main_window_examine_kpoint_convergence_launches_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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
        (2, 2, 2): main_window_module.SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
        (3, 3, 3): main_window_module.SweepPointRunState(status="completed", total_energy=-10.02, max_force=0.15),
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
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_adopt_warn.mufinder", "KPointAdoptWarn")
    window._document = doc

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_kpoint_default_adopted(8.0, "delta-threshold", 0.001, 0.05, {})

    assert warnings == ["Adopt failed"]
    window.close()


def test_main_window_on_kpoint_default_adopted_sets_project_default(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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
        "delta-threshold",
        0.001,
        0.05,
        {8.0: "Grid: 2 x 2 x 2"},
    )

    assert doc.project.recommended_kpoint_grid == (2, 2, 2)
    assert doc.project.recommended_kpoint_method == "delta-threshold"
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


def test_main_window_convergence_overview_restores_pinned_snapshots_on_reload(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
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


@pytest.mark.parametrize(
    "singlepoint_status",
    [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED],
)
def test_quantum_can_reset_to_stage_one_for_finished_bare_singlepoint(
    qapp, tmp_path: Path, singlepoint_status: JobStatus
) -> None:
    # A stage-1 failure with no phonon job used to be refused here, which left
    # the row permanently stuck: its status read "failed" so Submit was disabled
    # and Reset said "no resettable rows".
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / f"quantum_gate_{singlepoint_status.value}.mufinder", "QuantumGate"
    )
    window._document = doc
    definition, candidate, _ = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=singlepoint_status
    )

    assert window._quantum_can_reset_to_stage_one(definition, candidate) is True
    window.close()


@pytest.mark.parametrize("singlepoint_status", [JobStatus.DEFINED, JobStatus.RUNNING, JobStatus.QUEUED])
def test_quantum_cannot_reset_to_stage_one_for_unfinished_bare_singlepoint(
    qapp, tmp_path: Path, singlepoint_status: JobStatus
) -> None:
    # A merely-staged or still-running stage 1 has nothing to rewind, and
    # deleting a live job's records would strand the run.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / f"quantum_gate_{singlepoint_status.value}.mufinder", "QuantumGate"
    )
    window._document = doc
    definition, candidate, _ = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=singlepoint_status
    )

    assert window._quantum_can_reset_to_stage_one(definition, candidate) is False
    window.close()


def test_reset_quantum_corrections_to_stage_one_recovers_a_failed_singlepoint(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
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

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda self: main_window_module.QMessageBox.StandardButton.Yes,
    )

    window._on_reset_quantum_corrections_to_stage_one([candidate.key])

    assert singlepoint_job.id not in doc.jobs
    assert definition.parameters["singlepoint_job_ids_by_label"] == {}
    assert output_dir.exists() is False

    refreshed_candidate = window._quantum_source_candidate_by_key(candidate.key)
    assert refreshed_candidate is not None
    stage_state = window._quantum_stage_state(definition, refreshed_candidate)
    assert stage_state.stage == "singlepoint"
    assert stage_state.status == "ready"
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


def test_main_window_batch_row_payload_keeps_a_rows_own_untouched_mixed_values(qapp) -> None:
    from types import SimpleNamespace

    window = MainWindow()
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


def test_main_window_kpoint_examiner_and_adoption_use_the_sweep_supercell(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep run on a supercell is normalized and adopted against that supercell.

    The examiner's per-atom energies need the supercell's atom count and its
    spacings the supercell's lattice; the adopted grid must be stored with the
    supercell lattice, or later workflows would apply it unscaled to other cells.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_supercell_sweep.mufinder", "KPointSupercellSweep")
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
        fractional_positions=[[0.0, 0.0, 0.0], [0.125, 0.25, 0.25], [0.5, 0.0, 0.0], [0.625, 0.25, 0.25]],
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
        (1, 2, 2): main_window_module.SweepPointRunState(status="completed", total_energy=-20.0, max_force=0.2),
        (2, 4, 4): main_window_module.SweepPointRunState(status="completed", total_energy=-20.02, max_force=0.1),
    }
    monkeypatch.setattr(window._kpoint_convergence_controller, "build_states", lambda: states)
    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, cutoff_data, *, num_atoms, sweep_kind, point_labels, lattice_vectors=None,
                     initial_state=None, notes=(), parent=None) -> None:
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
    window._on_kpoint_default_adopted(16.0, "delta-threshold", 0.001, 0.05, {16.0: "Grid: 1 x 4 x 4"})

    assert doc.project.recommended_kpoint_grid == (1, 4, 4)
    assert doc.project.recommended_kpoint_cell_id == str(supercell.id)
    assert doc.project.recommended_kpoint_lattice == supercell.lattice_vectors
    window.close()


def test_kpoint_examiner_points_keep_one_grid_per_count_and_explain_the_rest() -> None:
    """Grids sharing a k-point count cannot share the examiner's axis."""

    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]]
    completed = main_window_module.SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.1)
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


def test_main_window_kpoint_overview_recomputes_spacings_on_the_sweep_cell(qapp, tmp_path: Path) -> None:
    """The overview refreshes stored spacings from the cell the snapshot was swept on."""

    from mufinder.science.kpoint_grids import spacing_from_grid

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_overview_supercell.mufinder", "KPointOverviewSupercell")
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
            "plot_data": {"cutoffs": [16.0], "energies": [-20.0], "point_spacings": {"16.0": 999.0}},
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
