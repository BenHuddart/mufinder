"""Behavior tests for main-window command flows and error handling."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QTreeWidgetItem

import mufinder.app.convergence_widget as convergence_widget_module
import mufinder.app.main_window as main_window_module
import mufinder.services.job_artifacts as job_artifacts_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app import workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES
from mufinder.castep_io.output_parser import CastepResult
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.cluster_setup import ClusterConfig
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, ComparisonTable, Job, MuonSite, Result, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.project_service import ProjectService
from mufinder.services.ssh_client import SSHAuthenticationError, SSHCredentials
from tests.support.reset_menu import install_fake_menu, open_reset_menu
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE


def _find_item_by_route(parent: QTreeWidgetItem, route: str) -> QTreeWidgetItem | None:
    if parent.data(0, 0x0100) == route:  # Qt.UserRole
        return parent
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child is None:
            continue
        found = _find_item_by_route(child, route)
        if found is not None:
            return found
    return None


def _select_route(window: MainWindow, route: str) -> None:
    item = None
    for index in range(window._project_tree.topLevelItemCount()):
        top_item = window._project_tree.topLevelItem(index)
        if top_item is None:
            continue
        item = _find_item_by_route(top_item, route)
        if item is not None:
            break
    assert item is not None
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()


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

    _select_route(window, "host_structure")
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

    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._STATUS_COLUMN).text() == "completed"
    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._TOTAL_ENERGY_COLUMN).text() == "-12.3456"
    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._MAX_FORCE_COLUMN).text() == ""
    assert window._convergence_view._cutoff_table.item(1, window._convergence_view._STATUS_COLUMN).text() == "ready"
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

    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._MAX_FORCE_COLUMN).text() == "0.0987"
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

    resolved_job = window._job_for_definition_cutoff(definition, 300.0)

    assert resolved_job is job
    assert window._definition_output_directory(definition) == doc.project_root / "files" / "convergence_cutoff"
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


def test_main_window_submit_jobs_locally_creates_run_and_calls_runner(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_submit_ui.mufinder", "LocalSubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs),
    )

    window._submit_jobs_locally([job])

    assert job.status == JobStatus.SUBMITTING
    assert len(submit_calls) == 1
    assert submit_calls[0]["seed_name"] == job.label
    assert submit_calls[0]["job_dir"] == output_dir / job.label
    assert job.num_processes == 1
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert doc.provenance[-1].action == "submit_jobs"
    assert doc.provenance[-1].parameters["job_labels"] == [job.label]

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

    _select_route(window, "host_structure")
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
            journal_item = _find_item_by_route(item, "provenance_log")
        if journal_item is not None:
            break
    assert journal_item is not None
    assert journal_item.text(0) == "Journal"

    _select_route(window, "provenance_log")

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
    _select_route(window, "provenance_log")

    texts = [label.text() for label in window._journal_view.findChildren(main_window_module.QLabel)]
    expected_day = doc.provenance[-1].timestamp.astimezone().strftime("%A %d %B %Y")
    assert expected_day in texts
    assert "Host structure imported" in texts
    window.close()


def test_main_window_submit_jobs_locally_uses_definition_output_for_non_convergence_stage(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_submit_generic.mufinder", "LocalSubmitGeneric")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="muon_relax_site_1",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    output_root = doc.project_root / "files" / "muon_relaxation"
    (output_root / job.label).mkdir(parents=True, exist_ok=True)

    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
        parameters={
            "output_directory": str(output_root.relative_to(doc.project_root)),
            "job_labels": [job.label],
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition

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
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs))

    window._submit_jobs_locally([job])

    assert len(submit_calls) == 1
    assert submit_calls[0]["job_dir"] == output_root / job.label
    window.close()


def test_main_window_submit_jobs_locally_resets_execution_target_after_cluster_reset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_submit_resets_target.mufinder", "LocalSubmitResetsTarget")

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="my-cluster",
    )
    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    job.status = JobStatus.FAILED
    job.remote_job_id = "12345"
    job.remote_directory = "/remote/conv_ecut_300"
    service.reset_failed_job(job)
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

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: None)

    window._submit_jobs_locally([job])

    assert job.execution_target == "local"
    assert doc.provenance[-1].parameters["execution_target"] == "local"
    window.close()


def test_main_window_submit_jobs_locally_warns_if_castep_unconfigured(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_warn_no_exec.mufinder", "WarnNoExec")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="", mpirun_path=""),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._submit_jobs_locally([job])

    assert warnings == ["CASTEP executable missing"]
    window.close()


def test_main_window_submit_jobs_on_cluster_creates_run_and_calls_runner(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_submit_ui.mufinder", "ClusterSubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    connect_calls: list[object] = []
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None: connect_calls.append((config.name, password)),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert connect_calls == [("alpha", None)]
    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["seed_name"] == job.label
    assert job.status == JobStatus.QUEUED
    assert job.execution_target == "alpha"
    assert job.remote_job_id == "99123"
    assert job.remote_directory == f"/remote/work/{job.label}"
    assert job.num_processes == 16
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED

    reloaded = service.load_project(doc.project_root)
    persisted_job = reloaded.jobs[job.id]
    assert persisted_job.status == JobStatus.QUEUED
    assert persisted_job.remote_job_id == "99123"
    assert persisted_job.remote_directory == f"/remote/work/{job.label}"
    persisted_runs = [run for run in reloaded.runs.values() if run.job_id == job.id]
    assert len(persisted_runs) == 1
    assert persisted_runs[0].status == JobStatus.QUEUED
    assert reloaded.provenance[-1].parameters["jobs"] == [
        {
            "job_id": str(job.id),
            "label": job.label,
            "status": "queued",
            "execution_target": "alpha",
            "remote_job_id": "99123",
            "remote_directory": f"/remote/work/{job.label}",
            "num_processes": 16,
            "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        }
    ]
    window.close()


def test_main_window_submit_jobs_on_cluster_passes_jump_host_credentials(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_jump_submit_ui.mufinder", "ClusterJumpSubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        default_working_directory="/mnt/redwoodfs/glam-user",
        queue_system="slurm",
        cores_per_node=24,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="Redwood",
                working_directory="/mnt/redwoodfs/glam-user",
                username="glam-user",
                job_name_prefix="conv",
                num_processes=24,
                walltime="01:00:00",
            )

        def credentials(self) -> SSHCredentials:
            return SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")

    connect_calls: list[tuple[str, SSHCredentials | None, str | None]] = []
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None, credentials=None: connect_calls.append((config.name, credentials, password)),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submit_calls.append(kwargs) or "99123")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert len(connect_calls) == 1
    assert connect_calls[0][0] == "Redwood"
    assert connect_calls[0][1] == SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")
    assert connect_calls[0][2] is None
    assert len(submit_calls) == 1
    window.close()


def test_main_window_resubmits_failed_cluster_job_without_replacing_remote_files(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit_ui.mufinder", "ClusterResubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    job.status = JobStatus.FAILED
    job.remote_job_id = "old99123"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.error_message = "previous scheduler failure"
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("resubmission should reuse the stored remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("resubmission should not check for overwrite"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("resubmission should not delete remote files"),
    )
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submit_calls[0]["reuse_existing_files"] is True
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED
    window.close()


def test_main_window_resubmit_passes_updated_resources_to_runner(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit_resources.mufinder", "ClusterResubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    job.status = JobStatus.FAILED
    job.remote_job_id = "old99123"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.error_message = "previous scheduler failure"
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=32,
                walltime="168:00:00",
                mem_per_cpu_gb=8.0,
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("resubmission should reuse the stored remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("resubmission should not check for overwrite"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("resubmission should not delete remote files"),
    )
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["walltime"] == "168:00:00"
    assert submit_calls[0]["mem_per_cpu_gb"] == 8.0
    assert submit_calls[0]["num_processes"] == 32
    assert submit_calls[0]["reuse_existing_files"] is True
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    assert job.num_processes == 32
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED
    window.close()


def test_main_window_submit_jobs_on_cluster_prefills_from_last_submission_defaults(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_last.mufinder", "ClusterPrefill")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x",
        "username": "alice",
        "job_name_prefix": "kpoint",
        "num_processes": 32,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
    }
    window._document = doc

    alpha = ClusterConfig(
        name="SCARF",
        hostname="scarf.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/work/scarf/alice",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [alpha, redwood])

    window._submit_jobs_on_cluster([job])

    assert captured_kwargs["default_cluster_name"] == "Redwood"
    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.cluster_name == "Redwood"
    assert initial_config.working_directory == "/mnt/redwoodfs/alice/project_x"
    assert initial_config.num_processes == 32
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    window.close()


def test_main_window_submit_jobs_on_cluster_prefills_same_workflow_without_reset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_same_workflow.mufinder", "ClusterPrefillSame")
    job = service.add_job(
        doc,
        label="conv_ecut_350",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
        "num_processes": 40,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
        "workflow_type": "convergence_cutoff",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom"
    assert initial_config.job_name_prefix == "cutoff_scan"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "convergence_cutoff"
    window.close()


def test_main_window_submit_jobs_on_cluster_resets_workdir_and_prefix_on_workflow_switch(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_workflow_switch.mufinder", "ClusterPrefillSwitch")
    job = service.add_job(
        doc,
        label="pristine_relax_01",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
        "num_processes": 40,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
        "workflow_type": "convergence_cutoff",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == ""
    assert initial_config.job_name_prefix == "pristine_relax"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "pristine_relaxation"
    window.close()


def test_main_window_submit_uep_jobs_on_cluster_resets_stale_prefix_on_workflow_switch(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_uep_switch.mufinder", "ClusterPrefillUepSwitch")
    job = service.add_job(
        doc,
        label="uep_01",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
        "num_processes": 40,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
        "workflow_type": "convergence_cutoff",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == ""
    assert initial_config.job_name_prefix == "uep"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "uep"
    window.close()


def test_main_window_submit_jobs_on_cluster_persists_last_submission_defaults(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_persist_last.mufinder", "ClusterPersist")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="Redwood",
                working_directory="/mnt/redwoodfs/alice/project_y",
                username="alice",
                job_name_prefix="pristine_relax",
                num_processes=24,
                walltime="",
                queue_name="redwood-priority",
                memory_gb=12.0,
                comment="night-run",
            )

        def password(self) -> str:
            return ""

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: redwood)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert warnings == ["Cluster connection failed"]
    persisted = doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY]
    assert persisted["cluster_name"] == "Redwood"
    assert persisted["working_directory"] == "/mnt/redwoodfs/alice/project_y"
    assert persisted["username"] == "alice"
    assert persisted["job_name_prefix"] == "pristine_relax"
    assert persisted["num_processes"] == 24
    assert persisted["queue_name"] == "redwood-priority"
    assert persisted["memory_gb"] == 12.0
    assert persisted["comment"] == "night-run"
    assert persisted["workflow_type"] == "convergence_cutoff"
    window.close()


def test_main_window_submit_jobs_on_cluster_connection_failure_warns(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_connect_fail.mufinder", "ClusterConnectFail")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert warnings == ["Cluster connection failed"]
    window.close()


def test_main_window_submit_jobs_on_cluster_blocks_unsafe_remote_labels(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_unsafe_label.mufinder", "ClusterUnsafeLabel")
    job = service.add_job(
        doc,
        label="bad label",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert warnings == ["Unsafe job labels"]
    window.close()


def test_main_window_submit_jobs_on_cluster_existing_remote_dir_yes_deletes(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_existing_dir_yes.mufinder", "ClusterExistingDirYes")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "yes")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == ["/remote/work/conv_ecut_300"]
    window.close()


def test_main_window_submit_jobs_on_cluster_reuses_quantum_continuation_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_quantum_continuation.mufinder", "ClusterQuantumContinuation")
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
    definition.parameters["singlepoint_job_ids_by_label"] = {}
    definition.parameters["phonon_job_ids_by_label"] = {}
    definition.parameters["job_seed_names_by_label"] = {str(site.id): "quantum_siteA"}
    config = definition.parameters.setdefault("configuration", {})
    assert isinstance(config, dict)
    row_settings = config.setdefault("row_settings", {})
    assert isinstance(row_settings, dict)
    row_settings[str(site.id)] = {
        "parameter_values": {},
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
    singlepoint_job.execution_target = "alpha"
    singlepoint_job.remote_directory = "/remote/work/quantum_siteA"
    definition.parameters["singlepoint_job_ids_by_label"][str(site.id)] = str(singlepoint_job.id)

    phonon_job = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    definition.parameters["phonon_job_ids_by_label"][str(site.id)] = str(phonon_job.id)
    definition.parameters["job_ids_by_label"] = definition.parameters["phonon_job_ids_by_label"]

    quantum_dir = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    quantum_dir.mkdir(parents=True, exist_ok=True)

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="quantum",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )

    checked_paths: list[str] = []
    deleted: list[str] = []
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: checked_paths.append(path) or False,
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submit_calls.append(kwargs) or "99123")
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("continuation submissions should not prompt to delete the remote directory"),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([phonon_job])

    assert checked_paths == []
    assert deleted == []
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/quantum_siteA"
    assert submit_calls[0]["reuse_existing_files"] is False
    assert submit_calls[0]["task_farming_enabled"] is True
    assert submit_calls[0]["num_farms"] == 4
    assert phonon_job.remote_directory == "/remote/work/quantum_siteA"
    assert phonon_job.execution_target == "alpha"

    window.close()


_CONTINUATION_CLUSTER = ClusterConfig(
    name="alpha",
    hostname="alpha.example",
    username="alice",
    auth_method="key",
    key_path="/tmp/key",
    default_working_directory="/remote/work",
    queue_system="slurm",
    cores_per_node=64,
    default_walltime="01:00:00",
    template_script_path="/templates/slurm.sh",
)


def _patch_continuation_cluster_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the cluster submission dialog with a fixed /remote/work config."""

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [_CONTINUATION_CLUSTER])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: _CONTINUATION_CLUSTER)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)


def _add_cluster_input_job(service: ProjectService, doc, label: str) -> Job:
    """Create a convergence job with its .cell/.param inputs on disk.

    The job directory basename equals the label, exactly as a real submission
    produces it, so ``_job_seed_name`` resolves the same seed from either the
    local directory or the remote basename.
    """

    job = service.add_job(
        doc,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_dir = doc.project_root / "files" / "convergence_cutoff" / label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{label}.cell").write_text("%BLOCK LATTICE_CART\n%ENDBLOCK LATTICE_CART\n", encoding="utf-8")
    (job_dir / f"{label}.param").write_text("TASK : GEOMETRYOPTIMIZATION\n", encoding="utf-8")
    return job


def _continue_cluster_job(service: ProjectService, job: Job, remote_directory: str) -> None:
    """Take a job through a failed cluster run and back to DEFINED for continuation."""

    job.execution_target = "alpha"
    job.remote_directory = remote_directory
    job.remote_job_id = "55501"
    job.status = JobStatus.FAILED
    service.prepare_job_continuation(job)
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == remote_directory


def _patch_continuation_cluster_runner(
    monkeypatch: pytest.MonkeyPatch,
    window: MainWindow,
    *,
    directory_exists: bool,
) -> dict[str, list]:
    """Patch the SSH-facing runner calls and record what the flow asked for."""

    recorded: dict[str, list] = {"checked": [], "deleted": [], "submitted": []}
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: recorded["checked"].append(path) or directory_exists,
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda path: recorded["deleted"].append(path),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: recorded["submitted"].append(kwargs) or "99123",
    )
    return recorded


def test_main_window_submit_jobs_on_cluster_reuses_continued_job_remote_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_continuation.mufinder", "ClusterContinuation")
    window._document = doc

    job = _add_cluster_input_job(service, doc, "conv_ecut_300")
    _continue_cluster_job(service, job, "/remote/work/conv_ecut_300")

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a continued job must never prompt to overwrite its own directory"),
    )

    window._submit_jobs_on_cluster([job])

    # The preserved directory is never probed, never offered for deletion, and
    # never deleted: it holds the .check file the continuation resumes from.
    assert recorded["checked"] == []
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submitted["allow_existing_remote_directory"] is True
    # Fresh .cell/.param are uploaded; only the reuse flag would suppress that.
    assert submitted["reuse_existing_files"] is False
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.execution_target == "alpha"

    window.close()


def test_main_window_submit_jobs_on_cluster_continuation_keeps_original_seed_name(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_continuation_seed.mufinder", "ClusterContinuationSeed")
    window._document = doc

    job = _add_cluster_input_job(service, doc, "conv_ecut_300")
    original_seed = window._job_seed_name(job)
    assert original_seed == "conv_ecut_300"
    _continue_cluster_job(service, job, "/remote/work/conv_ecut_300")

    # The remote basename now drives the seed, and must agree with the local one.
    assert window._job_seed_name(job) == original_seed

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a continued job must never prompt to overwrite its own directory"),
    )

    window._submit_jobs_on_cluster([job])

    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["seed_name"] == original_seed
    # The runner uploads {seed}.cell/{seed}.param from this directory, so a seed
    # drift would fail the submission outright.
    local_dir = Path(submitted["local_job_dir"])
    assert (local_dir / f"{original_seed}.cell").is_file()
    assert (local_dir / f"{original_seed}.param").is_file()

    window.close()


def test_main_window_submit_jobs_on_cluster_mixed_batch_checks_only_the_new_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mixed_batch.mufinder", "ClusterMixedBatch")
    window._document = doc

    continued = _add_cluster_input_job(service, doc, "conv_ecut_300")
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_300")
    fresh = _add_cluster_input_job(service, doc, "conv_ecut_400")
    assert fresh.status == JobStatus.DEFINED
    assert fresh.remote_directory is None

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=False)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("no directory exists, so nothing should prompt"),
    )

    window._submit_jobs_on_cluster([continued, fresh])

    # Exactly the fresh job's computed target is probed: the continued job is
    # excluded, the fresh job's existence check is unchanged.
    assert recorded["checked"] == ["/remote/work/conv_ecut_400"]
    assert recorded["deleted"] == []
    by_directory = {call["remote_directory"]: call for call in recorded["submitted"]}
    assert set(by_directory) == {"/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"}
    assert by_directory["/remote/work/conv_ecut_300"]["allow_existing_remote_directory"] is True
    assert by_directory["/remote/work/conv_ecut_400"]["allow_existing_remote_directory"] is False
    assert by_directory["/remote/work/conv_ecut_400"]["reuse_existing_files"] is False

    window.close()


def test_main_window_submit_jobs_on_cluster_continuation_survives_target_collision(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A colliding batch job must not drag a preserved directory into the prompt.

    Overwrite decisions are matched by target string, so a second job whose
    computed target equals a continued job's preserved directory would otherwise
    let one prompt answer skip or delete the continuation.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_collision.mufinder", "ClusterCollision")
    window._document = doc

    continued = _add_cluster_input_job(service, doc, "conv_ecut_300")
    collider = _add_cluster_input_job(service, doc, "conv_ecut_400")
    # The continued job was originally submitted into the directory the collider
    # now computes for itself.
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_400")

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    prompted: list[str] = []
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda label, _target: prompted.append(label) or "no",
    )

    window._submit_jobs_on_cluster([continued, collider])

    assert prompted == ["conv_ecut_400"]
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    assert recorded["submitted"][0]["seed_name"] == "conv_ecut_400"
    assert recorded["submitted"][0]["remote_directory"] == "/remote/work/conv_ecut_400"
    assert recorded["submitted"][0]["allow_existing_remote_directory"] is True
    assert continued.status == JobStatus.QUEUED
    assert collider.status == JobStatus.DEFINED

    window.close()


def test_main_window_submit_jobs_on_cluster_collision_overwrite_spares_continued_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering "yes" for a colliding job must not delete a preserved directory.

    The overwrite decision belongs to the colliding job, but the deletion it
    authorises would land on the directory holding the continuation's .check file.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_collision_yes.mufinder", "ClusterCollisionYes")
    window._document = doc

    continued = _add_cluster_input_job(service, doc, "conv_ecut_300")
    collider = _add_cluster_input_job(service, doc, "conv_ecut_400")
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_400")

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "yes")

    window._submit_jobs_on_cluster([continued, collider])

    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 2
    assert continued.status == JobStatus.QUEUED
    assert collider.status == JobStatus.QUEUED

    window.close()


def test_main_window_submit_jobs_on_cluster_dashboard_resubmit_still_reuses_files(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The continuation branch must not capture the dashboard resubmit path.

    Dashboard resubmission passes FAILED/CANCELLED jobs, which the continuation
    branch ignores, and keeps its own reuse-the-remote-inputs semantics.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit.mufinder", "ClusterResubmit")
    window._document = doc

    job = _add_cluster_input_job(service, doc, "conv_ecut_300")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.status = JobStatus.FAILED

    _patch_continuation_cluster_dialog(monkeypatch)
    recorded = _patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a resubmission into its own directory must not prompt"),
    )

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.FAILED, JobStatus.CANCELLED},
        reuse_existing_remote_files=True,
    )

    assert recorded["checked"] == []
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submitted["reuse_existing_files"] is True
    assert submitted["allow_existing_remote_directory"] is False

    window.close()


def test_main_window_submit_jobs_on_cluster_yes_to_all_requires_confirmation(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_yes_all_confirm.mufinder", "ClusterYesAllConfirm")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    jobs: list[Job] = []
    for cutoff in (300, 400):
        job = service.add_job(
            doc,
            label=f"conv_ecut_{cutoff}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        jobs.append(job)
    output_dir = doc.project_root / "files" / "convergence_cutoff"
    for job in jobs:
        (output_dir / job.label).mkdir(parents=True, exist_ok=True)
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 400.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_dir,
        job_ids_by_cutoff={
            main_window_module.MainWindow._cutoff_key(300.0): str(jobs[0].id),
            main_window_module.MainWindow._cutoff_key(400.0): str(jobs[1].id),
        },
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    prompt_calls: list[str] = []
    confirm_calls: list[list[str]] = []
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: prompt_calls.append("prompt") or "yes_to_all",
    )
    monkeypatch.setattr(
        window,
        "_confirm_bulk_remote_delete",
        lambda paths: confirm_calls.append(list(paths)) or (len(paths) == 2),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster(jobs)

    assert prompt_calls == ["prompt"]
    assert len(confirm_calls) == 1
    assert sorted(deleted) == ["/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"]
    window.close()


def test_main_window_cluster_status_handlers_update_job_and_run(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_status_ui.mufinder", "ClusterStatus")
    job = service.add_job(
        doc,
        label="cluster_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    service.submit_job(job)
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.SUBMITTING)
    doc.runs[run.id] = run
    window._document = doc

    window._on_cluster_job_status_changed(job.id, JobStatus.QUEUED.value)
    assert job.status == JobStatus.QUEUED

    window._on_cluster_job_status_changed(job.id, JobStatus.RUNNING.value)
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING

    window._on_cluster_job_status_changed(job.id, JobStatus.RETRIEVING.value)
    assert job.status == JobStatus.RETRIEVING

    window._on_cluster_job_status_changed(job.id, JobStatus.STOPPED.value)
    assert job.status == JobStatus.STOPPED
    assert run.status == JobStatus.STOPPED
    assert run.completed_at is not None
    assert doc.provenance[-1].action == "job_stopped"

    window.close()


def test_dashboard_flush_is_coalesced_across_a_completion_burst(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "autosave_coalesce.mufinder", "AutosaveCoalesce")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    # A burst of lifecycle autosave requests does not write per request.
    window._request_dashboard_flush(failure_message="x")
    window._request_dashboard_flush(failure_message="x")
    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    # The flush (called after a queue poll / by the backstop timer) writes once.
    window._flush_dashboard()
    assert save_calls == ["save"]

    # Nothing pending -> no further writes.
    window._flush_dashboard()
    assert save_calls == ["save"]
    window.close()


def test_close_persists_pending_dashboard_flush(qapp, tmp_path: Path) -> None:
    # A coalesced flush pending at close must be persisted, not dropped — before
    # autosave coalescing these completions were already written to disk per job.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "close_flush.mufinder", "CloseFlush")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    window.close()
    assert save_calls == ["save"]


def test_cluster_queue_checked_schedules_deferred_flush(qapp, tmp_path: Path) -> None:
    # _on_cluster_queue_checked used to call _flush_dashboard() directly,
    # bypassing the debounce/rate-limit entirely. It must now route through
    # _request_dashboard_flush like every other lifecycle handler: the queue
    # check drains any already-pending flush, but does not write synchronously.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "queue_flush.mufinder", "QueueFlush")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
    # Not synchronous: the write is still only scheduled.
    assert save_calls == []
    assert window._dashboard_flush_pending is True

    # The metadata stamp landed in the in-memory document regardless of when
    # the flush itself runs.
    window._flush_dashboard()
    assert save_calls == ["save"]
    window.close()


def test_cluster_queue_checked_with_no_activity_does_not_schedule_flush(
    qapp, tmp_path: Path
) -> None:
    # A routine queue poll that stamps nothing and has no pending completion
    # must not start scheduling flushes on its own — only actual job activity
    # (via mark_queue_checked stamping a job, or a completion handler already
    # having marked pending) should arm the timer.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "queue_no_activity.mufinder", "QueueNoActivity")
    window._document = doc

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
    assert window._dashboard_flush_pending is False
    assert window._dashboard_flush_timer.isActive() is False
    window.close()


def test_dashboard_flush_isolated_event_uses_short_debounce(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An isolated lifecycle event (nothing nearby) should still arm the short
    # debounce, not the longer min-flush-interval floor, preserving the
    # existing prompt-flush feel.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "isolated_event.mufinder", "IsolatedEvent")
    window._document = doc

    clock = [1_000.0]
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: clock[0])

    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_timer.isActive() is True
    assert window._dashboard_flush_timer.interval() == window._DASHBOARD_FLUSH_DEBOUNCE_MS
    window.close()


def test_dashboard_flush_stream_is_rate_limited_to_min_interval(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A steady stream of requests spaced closer than the min flush interval
    # (but further apart than the debounce) must flush on a bounded ~2s
    # cadence during the stream, not only once the stream stops, and must not
    # have each new request push the deadline further into the future.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "flush_stream.mufinder", "FlushStream")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    clock = [0.0]
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: clock[0])

    # First request: isolated so far -> short debounce deadline at t=0.3s.
    window._request_dashboard_flush(failure_message="x")
    first_deadline = window._dashboard_flush_deadline
    assert first_deadline == pytest.approx(0.3)

    # Fire the debounce timer as Qt would once its interval elapses.
    clock[0] = 0.3
    window._dashboard_flush_timer.timeout.emit()
    assert save_calls == ["save"]
    assert window._dashboard_last_flush_at == pytest.approx(0.3)

    # A request 0.5s after the last flush: within the 2s floor, so the
    # deadline should land 2s after the last flush (t=2.3s), not 0.3s later.
    clock[0] = 0.8
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    # Further requests inside the same armed window must not push the
    # deadline further out.
    clock[0] = 1.3
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    clock[0] = 1.8
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    # The armed deadline fires on schedule, bounding the burst to one flush
    # per ~2s rather than starving until the stream ends.
    clock[0] = 2.3
    window._dashboard_flush_timer.timeout.emit()
    assert save_calls == ["save", "save"]
    assert window._dashboard_last_flush_at == pytest.approx(2.3)

    # Trailing request after the stream ends still schedules a guaranteed
    # final flush.
    clock[0] = 2.5
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_pending is True
    window._flush_dashboard()
    assert save_calls == ["save", "save", "save"]
    window.close()


def test_main_window_cluster_connection_indicator_and_refresh(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    monkeypatch.setattr(
        type(window._cluster_runner),
        "connected_cluster_name",
        property(lambda _self: "alpha"),
    )

    window._on_cluster_connection_status_changed(True)
    assert window._cluster_connection_label.text() == "Cluster: connected (alpha)"
    assert window._cluster_refresh_button.isEnabled() is True

    window._on_cluster_queue_checked("12:34:56")
    assert "12:34:56" in window._cluster_queue_checked_label.text()

    calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: calls.append("refresh"))
    window._on_refresh_cluster_queue()
    assert calls == ["refresh"]

    window._on_cluster_connection_status_changed(False)
    assert window._cluster_connection_label.text() == "Cluster: disconnected"
    assert window._cluster_refresh_button.isEnabled() is True
    window.close()


def test_main_window_cluster_queue_checked_formats_timezone_aware_times_locally(qapp) -> None:
    window = MainWindow()

    class _LocalTime:
        def strftime(self, _format: str) -> str:
            return "12:34:56"

    class _AwareTime:
        tzinfo = object()

        def astimezone(self):
            return _LocalTime()

        def strftime(self, _format: str) -> str:
            return "11:34:56"

    window._on_cluster_queue_checked(_AwareTime())

    assert "12:34:56" in window._cluster_queue_checked_label.text()
    window.close()


def test_main_window_cluster_queue_checked_uses_snapshot_timestamp(qapp) -> None:
    window = MainWindow()

    snapshot = SimpleNamespace(
        checked_at=datetime(2026, 4, 16, 12, 34, 56, tzinfo=UTC),
        elapsed_by_job_id={},
    )

    window._on_cluster_queue_checked(snapshot)

    expected = snapshot.checked_at.astimezone().strftime("%H:%M:%S")
    assert window._cluster_queue_checked_label.text() == f"Queue last checked: {expected}"
    window.close()


def test_main_window_cluster_queue_checked_unknown_object_stays_compact(qapp) -> None:
    window = MainWindow()

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=object(), elapsed_by_job_id={}))

    assert window._cluster_queue_checked_label.text() == "Queue last checked: unknown"
    window.close()


def test_main_window_runtime_column_updates_for_running_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "runtime_tick.mufinder", "RuntimeTick")

    job = service.add_job(
        doc,
        label="uep_runtime_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC),
    )
    window._document = doc
    window._refresh_jobs_table()

    assert window._job_runtime_timer.isActive() is True
    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "00:00"

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC),
    )
    window._refresh_active_job_runtime_cells()

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "01:05"

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC)
    window._refresh_active_job_runtime_cells()

    assert window._job_runtime_timer.isActive() is False
    window.close()


def test_main_window_cluster_runtime_uses_queue_elapsed_and_ticks_between_polls(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_runtime_tick.mufinder", "ClusterRuntimeTick")

    job = service.add_job(
        doc,
        label="uep_cluster_runtime_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.RUNNING
    job.execution_target = "redwood"

    window._document = doc
    window._refresh_jobs_table()

    snapshot_time = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(main_window_module, "utc_now", lambda: snapshot_time)
    window._on_cluster_queue_checked(
        SimpleNamespace(
            checked_at=snapshot_time,
            elapsed_by_job_id={job.id: "00:12:34"},
        )
    )

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "12:34"
    assert window._job_runtime_timer.isActive() is True

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 4, tzinfo=UTC),
    )
    window._refresh_active_job_runtime_cells()

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "13:38"
    window.close()


def test_main_window_runtime_timer_pauses_while_job_dock_hidden(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dock_pause.mufinder", "DockPause")

    job = service.add_job(
        doc,
        label="uep_dock_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        main_window_module, "utc_now", lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    )

    window._document = doc
    window._refresh_jobs_table()
    assert window._job_runtime_timer.isActive() is True

    # Dock becoming not-visible (closed, obscured, or window minimized — all of
    # which emit visibilityChanged(False)) should pause the per-second updates.
    window._on_jobs_dock_visibility_changed(False)
    assert window._jobs_dock_visible is False
    assert window._job_runtime_timer.isActive() is False

    # A live-runtime resync while not visible must not re-arm the timer.
    window._sync_job_runtime_timer()
    assert window._job_runtime_timer.isActive() is False

    # Becoming visible again re-arms it and refreshes the (now stale) cells.
    window._on_jobs_dock_visibility_changed(True)
    assert window._jobs_dock_visible is True
    assert window._job_runtime_timer.isActive() is True

    window.close()


def test_main_window_tab_navigation_does_not_recompute_stage_icons(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "nav_icons.mufinder", "NavIcons")
    window._document = doc
    window._refresh_ui_after_project_change()

    # Switching tabs cannot change workflow completion, so navigation must not
    # re-run the (job/result-scanning) completion checks.
    calls = 0
    original = window._refresh_workflow_stage_icons

    def _counting() -> None:
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(window, "_refresh_workflow_stage_icons", _counting)

    item = QTreeWidgetItem(["Candidates"])
    item.setData(0, 0x0100, "muon_candidates")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._muon_generation_view
    assert calls == 0

    window.close()


def test_main_window_refresh_ui_seeds_cluster_tracking_for_active_remote_jobs(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_tracking_seed.mufinder", "ClusterTrackingSeed")

    tracked_job = service.add_job(
        doc,
        label="kpoint_convergence_kgrid_15x15x15",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
        execution_target="Redwood",
    )
    tracked_job.status = JobStatus.RUNNING
    tracked_job.remote_job_id = "661459"
    tracked_job.remote_directory = "/mnt/redwoodfs/alice/project/kpoint_convergence_kgrid_15x15x15"

    local_job = service.add_job(
        doc,
        label="local_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
        execution_target="local",
    )
    local_job.status = JobStatus.RUNNING

    job_dir = doc.project_root / "files" / tracked_job.workflow_stage.value / tracked_job.label
    job_dir.mkdir(parents=True, exist_ok=True)

    clear_calls: list[str] = []
    track_calls: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner, "clear_tracked_jobs", lambda: clear_calls.append("clear"))
    monkeypatch.setattr(
        window._cluster_runner,
        "track_existing_job",
        lambda **kwargs: track_calls.append(kwargs),
    )

    window._document = doc
    window._refresh_ui_after_project_change()

    assert clear_calls == ["clear"]
    assert len(track_calls) == 1
    assert track_calls[0]["job_id"] == tracked_job.id
    assert track_calls[0]["remote_job_id"] == "661459"
    assert track_calls[0]["cluster_name"] == "Redwood"
    assert track_calls[0]["status"] == JobStatus.RUNNING
    window.close()


def test_main_window_disconnected_refresh_attempts_active_cluster_targets(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_refresh_reconnect.mufinder", "ClusterRefresh")

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.RUNNING

    scarf_job = service.add_job(
        doc,
        label="scarf_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="SCARF",
    )
    scarf_job.status = JobStatus.QUEUED

    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    scarf = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="builtin:scarf-slurm",
    )

    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood, scarf])

    connect_calls: list[str] = []
    refresh_calls: list[str] = []

    def _connect(config, password=None):
        _ = password
        connect_calls.append(config.name)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh"))

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == ["Redwood", "SCARF"]
    assert refresh_calls == ["refresh", "refresh"]
    assert warnings == []
    window.close()


def test_main_window_disconnected_refresh_prompts_password_for_password_cluster(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_refresh_password_prompt.mufinder", "ClusterRefreshPwd")

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.QUEUED
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="password",
        key_path="",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise RuntimeError("Password is required for password authentication.")

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh"))
    monkeypatch.setattr(
        main_window_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == [("Redwood", None), ("Redwood", "secret")]
    assert refresh_calls == ["refresh"]
    assert warnings == []
    window.close()


def test_main_window_disconnected_refresh_prompts_passphrase_for_key_cluster(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_refresh_key_prompt.mufinder", "ClusterRefreshKey")

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.QUEUED
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        key_path="/Users/alice/.ssh/id_ed25519",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise SSHAuthenticationError("SSH key authentication failed: private key is encrypted")

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh"))
    monkeypatch.setattr(
        main_window_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret-passphrase"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == [("Redwood", None), ("Redwood", "secret-passphrase")]
    assert refresh_calls == ["refresh"]
    assert warnings == []
    window.close()


def test_main_window_cluster_poll_failed_shows_detailed_warning(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._on_cluster_poll_failed("Queue polling failed for alpha: timeout")

    assert warnings == ["Queue polling failed"]
    window.close()


def test_main_window_connect_with_trust_handles_jump_and_target_host_keys(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    config = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )
    credentials = SSHCredentials(target_secret="glam", jump_secret="phys")
    prompts: list[str] = []
    connect_attempts: list[int] = []

    def _connect(_config, *, credentials=None, password=None):
        _ = credentials, password
        connect_attempts.append(len(connect_attempts) + 1)
        if len(connect_attempts) == 1:
            raise main_window_module.SSHUnknownHostKeyError(
                hostname="bastion.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:bastion",
                key_base64="YmFzdGlvbg==",
                endpoint="jump",
            )
        if len(connect_attempts) == 2:
            raise main_window_module.SSHUnknownHostKeyError(
                hostname="glamdring.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:glamdring",
                key_base64="Z2xhbWRyaW5n",
                endpoint="target",
            )

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        main_window_module,
        "prompt_to_trust_host_key",
        lambda _parent, _config, exc: prompts.append(exc.endpoint) or True,
    )

    window._connect_to_cluster_with_trust(config, credentials=credentials)

    assert connect_attempts == [1, 2, 3]
    assert prompts == ["jump", "target"]
    assert (
        window._settings.value(
            "trusted_host_keys/Redwood/jump/bastion.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    assert (
        window._settings.value(
            "trusted_host_keys/Redwood/glamdring.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    window.close()


def test_main_window_remote_directory_prompt_mappings(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    sequence = iter(
        [
            main_window_module.QMessageBox.StandardButton.Yes,
            main_window_module.QMessageBox.StandardButton.YesToAll,
            main_window_module.QMessageBox.StandardButton.NoToAll,
            main_window_module.QMessageBox.StandardButton.No,
        ]
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "exec", lambda _self: next(sequence))

    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "yes"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "yes_to_all"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "no_to_all"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "no"
    window.close()


def test_main_window_confirm_bulk_remote_delete_branches(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    assert window._confirm_bulk_remote_delete([]) is True

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda _self: main_window_module.QMessageBox.StandardButton.Yes,
    )
    assert window._confirm_bulk_remote_delete(["/b", "/a", "/a"]) is True

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda _self: main_window_module.QMessageBox.StandardButton.No,
    )
    assert window._confirm_bulk_remote_delete(["/a"]) is False
    window.close()


def test_main_window_submit_jobs_on_cluster_no_to_all_skips_deletion(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_no_all_skip.mufinder", "ClusterNoAll")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submitted.append(kwargs) or "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "no_to_all")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == []
    assert len(submitted) == 0
    assert job.status == JobStatus.DEFINED
    window.close()


def test_main_window_submit_jobs_on_cluster_no_skips_submission(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_no_skip.mufinder", "ClusterNo")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submitted.append(kwargs) or "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "no")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == []
    assert len(submitted) == 0
    assert job.status == JobStatus.DEFINED
    window.close()


def test_main_window_submit_jobs_on_cluster_reuses_connection_without_success_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_reuse_conn.mufinder", "ClusterReuse")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        type(window._cluster_runner),
        "is_connected",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        type(window._cluster_runner),
        "connected_cluster_name",
        property(lambda _self: "alpha"),
    )
    monkeypatch.setattr(
        type(window._cluster_runner),
        "connected_username",
        property(lambda _self: "alice"),
    )

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert "Cluster connection successful" not in infos
    window.close()


def test_main_window_submit_from_dashboard_dispatches_local_and_cluster(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_dashboard_mix.mufinder", "SubmitDashboardMix")

    local_job = service.add_job(
        doc,
        label="local_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="local",
    )
    cluster_job = service.add_job(
        doc,
        label="cluster_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    window._document = doc

    local_calls: list[list[Job]] = []
    cluster_calls: list[tuple[list[Job], str | None]] = []
    monkeypatch.setattr(window, "_submit_jobs_locally", lambda jobs: local_calls.append(jobs))
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, default_cluster_name=None: cluster_calls.append((jobs, default_cluster_name)),
    )

    window._on_submit_from_dashboard([local_job.id, cluster_job.id])

    assert local_calls == [[local_job]]
    assert cluster_calls == [([cluster_job], "alpha")]
    window.close()


def test_main_window_submit_jobs_locally_warns_missing_output_paths(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_missing_output.mufinder", "SubmitMissingOutput")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
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
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._submit_jobs_locally([job])

    assert "Missing job directories" in warnings
    window.close()


def test_main_window_cancel_jobs_only_active_states(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cancel_local.mufinder", "CancelLocal")
    active_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    inactive_job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    active_job.status = JobStatus.RUNNING
    inactive_job.status = JobStatus.DEFINED
    window._document = doc

    cancelled_ids: list[object] = []
    monkeypatch.setattr(window._local_runner, "cancel", lambda job_id: cancelled_ids.append(job_id))

    window._cancel_jobs([active_job, inactive_job])

    assert cancelled_ids == [active_job.id]
    window.close()


def test_main_window_run_lookup_prefers_active_mapping(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "run_lookup.mufinder", "RunLookup")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    run_1 = CalculationRun(job_id=job.id, run_index=1)
    run_2 = CalculationRun(job_id=job.id, run_index=2)
    doc.runs[run_1.id] = run_1
    doc.runs[run_2.id] = run_2
    window._document = doc

    window._active_run_by_job[job.id] = run_1.id
    assert window._run_for_job(job.id) is run_1

    window._active_run_by_job.clear()
    assert window._run_for_job(job.id) is run_2
    window.close()


def test_main_window_local_status_handlers_update_job_and_run(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_statuses.mufinder", "LocalStatuses")
    job = service.add_job(
        doc,
        label="job_state",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.SUBMITTING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id
    job.status = JobStatus.SUBMITTING

    window._on_local_job_status_changed(job.id, JobStatus.QUEUED.value)
    assert job.status == JobStatus.QUEUED
    assert run.status == JobStatus.QUEUED

    window._on_local_job_started(job.id)
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None

    window._on_local_job_failed(job.id, "boom")
    assert job.status == JobStatus.FAILED
    assert run.status == JobStatus.FAILED
    assert doc.provenance[-1].action == "job_failed"

    job.status = JobStatus.RUNNING
    run.status = JobStatus.RUNNING
    window._on_local_job_cancelled(job.id)
    assert job.status == JobStatus.CANCELLED
    assert run.status == JobStatus.CANCELLED
    assert doc.provenance[-1].action == "job_cancelled"
    window.close()


def test_main_window_local_finish_creates_result_and_completes_job(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_finish.mufinder", "LocalFinish")
    job = service.add_job(
        doc,
        label="job_finish",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    output_dir = doc.project_root / "files" / "muon_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / f"{job.label}.castep").write_text("castep", encoding="utf-8")

    parsed = CastepResult(
        completed=True,
        total_energy=-10.5,
        warnings=("warning",),
        geom_converged=True,
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert run.status == JobStatus.COMPLETED
    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["total_energy"] == -10.5
    assert any(path.endswith(f"{job.label}.castep") for path in result.output_files)
    assert window._jobs_table.rowCount() == 0
    assert doc.provenance[-1].action == "job_completed"
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
    """The shared out.cell prologue in _on_local_job_finished must fail a
    pristine job whose relaxed structure output is absent, BEFORE the migrated
    PristineRelaxationController.build_result is ever reached. This pins the
    inline failure-detection guard that build_result deliberately does not own
    (ControllerResult has no failure channel)."""

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
    would happily validate it as this run's geometry. The inline stale-output
    guard fails the job instead, so no result is built from a superseded file."""

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

    assert main_window_module._output_predates_run(path, None) is False
    assert main_window_module._output_predates_run(None, written) is False
    # Written after the run started, and written a hair before it: both fresh.
    assert main_window_module._output_predates_run(
        path, written - timedelta(minutes=5)
    ) is False
    assert main_window_module._output_predates_run(
        path, written + timedelta(seconds=1)
    ) is False
    # A whole run older: stale.
    assert main_window_module._output_predates_run(
        path, written + timedelta(minutes=5)
    ) is True
    # An unreadable path is not evidence of staleness either.
    assert main_window_module._output_predates_run(tmp_path / "absent", written) is False


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


def test_main_window_job_output_directory_fallback_by_stage(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "output_fallback.mufinder", "OutputFallback")
    job = service.add_job(
        doc,
        label="job_path",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    window._document = doc

    fallback = doc.project_root / "files" / "muon_relaxation" / job.label
    fallback.mkdir(parents=True, exist_ok=True)

    resolved = window._job_output_directory(job)

    assert resolved == fallback
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
        calc_type=main_window_module.MUON_RELAXATION_CALC_TYPES["positive_muon"],
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
    assert window._active_muon_relaxation_definition().id == newer_definition.id

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


def test_main_window_dashboard_uses_row_buttons_and_no_context_menu(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_buttons.mufinder", "JobsButtons")
    service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    completed_job = service.add_job(
        doc,
        label="job_completed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    cancelled_job = service.add_job(
        doc,
        label="job_cancelled",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING
    failed_job.status = JobStatus.FAILED
    completed_job.status = JobStatus.COMPLETED
    cancelled_job.status = JobStatus.CANCELLED
    window._document = doc
    resubmitted: list[list[object]] = []
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(window, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids)))
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    monkeypatch.setattr(window, "_on_inspect_failed_from_dashboard", lambda ids: inspected.append(list(ids)))
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: reset.append(list(ids)))

    window._refresh_jobs_table()

    assert window._jobs_table.contextMenuPolicy() == main_window_module.Qt.ContextMenuPolicy.CustomContextMenu
    assert window._jobs_table.rowCount() == 3

    running_resubmit_button = window._jobs_table.cellWidget(0, window._JOB_SUBMIT_COLUMN)
    failed_resubmit_button = window._jobs_table.cellWidget(1, window._JOB_SUBMIT_COLUMN)
    cancelled_resubmit_button = window._jobs_table.cellWidget(2, window._JOB_SUBMIT_COLUMN)
    cancel_button = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    inspect_button = window._jobs_table.cellWidget(1, window._JOB_INSPECT_COLUMN)
    reset_button = window._jobs_table.cellWidget(1, window._JOB_RESET_COLUMN)
    cancelled_reset_button = window._jobs_table.cellWidget(2, window._JOB_RESET_COLUMN)

    assert isinstance(running_resubmit_button, main_window_module.QPushButton)
    assert isinstance(failed_resubmit_button, main_window_module.QPushButton)
    assert isinstance(cancelled_resubmit_button, main_window_module.QPushButton)
    assert isinstance(cancel_button, main_window_module.QPushButton)
    assert isinstance(inspect_button, main_window_module.QPushButton)
    assert isinstance(reset_button, main_window_module.QPushButton)
    assert isinstance(cancelled_reset_button, main_window_module.QPushButton)

    assert running_resubmit_button.text() == "Resubmit"
    assert failed_resubmit_button.text() == "Resubmit"
    assert cancelled_resubmit_button.text() == "Resubmit"
    assert running_resubmit_button.isEnabled() is False
    assert failed_resubmit_button.isEnabled() is True
    assert cancelled_resubmit_button.isEnabled() is True
    assert window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN).isEnabled() is True
    assert cancel_button.isEnabled() is True
    assert inspect_button.isEnabled() is True
    assert reset_button.isEnabled() is True
    assert cancelled_reset_button.isEnabled() is True

    failed_resubmit_button.click()
    cancelled_resubmit_button.click()
    cancel_button.click()
    inspect_button.click()
    reset_button.click()

    assert resubmitted == [[failed_job.id], [cancelled_job.id]]
    assert cancelled == [[running_job.id]]
    assert inspected == [[failed_job.id]]
    assert reset == [[failed_job.id]]
    window.close()


def test_main_window_dashboard_cancel_applies_to_snapshot_rows(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the clicked row is in a multi-row snapshot, prompt and apply to all selected rows."""
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel.mufinder", "SnapCancel")
    job0 = service.add_job(doc, label="j0", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job1 = service.add_job(doc, label="j1", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job0.status = JobStatus.RUNNING
    job1.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.ALL_SELECTED)
    window._refresh_jobs_table()

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0, 1)
    cancel_btn.clicked.emit()

    assert len(cancelled) == 1
    assert set(cancelled[0]) == {job0.id, job1.id}
    window.close()


def test_main_window_dashboard_cancel_single_row_no_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel_single.mufinder", "SnapCancelSingle")
    job0 = service.add_job(doc, label="j0", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job0.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    prompted: list[bool] = []
    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED)
    window._refresh_jobs_table()

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0,)
    cancel_btn.clicked.emit()

    assert cancelled == [[job0.id]]
    assert prompted == []
    window.close()


def test_main_window_dashboard_cancel_unselected_row_no_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel_unsel.mufinder", "SnapCancelUnsel")
    job0 = service.add_job(doc, label="j0", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job1 = service.add_job(doc, label="j1", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job0.status = JobStatus.RUNNING
    job1.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    prompted: list[bool] = []
    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED)
    window._refresh_jobs_table()

    # Snapshot only contains row 0; click Cancel on row 1 (not in snapshot)
    cancel_btn = window._jobs_table.cellWidget(1, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0,)
    cancel_btn.clicked.emit()

    assert cancelled == [[job1.id]]
    assert prompted == []
    window.close()


def test_main_window_dashboard_status_uses_color_cues(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_status_colors.mufinder", "JobsStatusColors")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc
    window._refresh_jobs_table()

    status_item = window._jobs_table.item(0, window._JOB_STATUS_COLUMN)
    assert status_item is not None
    assert status_item.foreground().color().name() == "#b91c1c"
    window.close()


def test_main_window_dashboard_shows_job_process_count(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_process_count.mufinder", "JobsProcessCount")
    job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.RUNNING
    job.num_processes = 8
    window._document = doc

    window._refresh_jobs_table()

    processes_item = window._jobs_table.item(0, window._JOB_PROCESSES_COLUMN)
    assert processes_item is not None
    assert processes_item.text() == "8"
    window.close()


def test_main_window_dashboard_handlers_no_document_are_noops(qapp) -> None:
    window = MainWindow()

    window._on_submit_local_from_dashboard([uuid4()])
    window._on_resubmit_from_dashboard([uuid4()])
    window._on_cancel_from_dashboard([uuid4()])
    window._on_inspect_failed_from_dashboard([uuid4()])
    window._on_reset_failed_from_dashboard([uuid4()])

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


def test_main_window_dashboard_handlers_forward_selected_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_forward.mufinder", "DashboardForward")
    job = service.add_job(
        doc,
        label="job_dash",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    submitted: list[list[object]] = []
    resubmitted: list[tuple[list[object], dict[str, object]]] = []
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(window, "_submit_jobs_locally", lambda jobs: submitted.append(jobs))
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window, "_inspect_failed_jobs", lambda jobs: inspected.append(jobs))
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda jobs: reset.append(jobs))

    window._on_submit_local_from_dashboard([job.id])
    window._on_cancel_from_dashboard([job.id])
    window._on_inspect_failed_from_dashboard([job.id])
    window._on_reset_failed_from_dashboard([job.id])

    assert submitted == [[job]]
    assert cancelled == [[job]]
    assert inspected == [[job]]
    assert reset == [[job]]

    job.status = JobStatus.CANCELLED
    monkeypatch.setattr(
        window,
        "_submit_jobs_locally",
        lambda jobs, **kwargs: resubmitted.append((jobs, kwargs)),
    )
    window._on_resubmit_from_dashboard([job.id])

    assert resubmitted
    assert resubmitted[0][0] == [job]
    assert resubmitted[0][1]["eligible_statuses"] == {JobStatus.FAILED, JobStatus.CANCELLED}
    assert "cancelled" in resubmitted[0][1]["no_eligible_message"]
    window.close()



def _dashboard_with_jobs(window, service, doc, statuses):
    jobs = []
    for index, status in enumerate(statuses):
        job = service.add_job(
            doc,
            label=f"job_{index}_{status.value}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = status
        jobs.append(job)
    window._document = doc
    window._jobs_table.setRowCount(len(jobs))
    selection_model = window._jobs_table.selectionModel()
    for row, job in enumerate(jobs):
        label_item = main_window_module.QTableWidgetItem(job.label)
        label_item.setData(main_window_module.Qt.ItemDataRole.UserRole, str(job.id))
        window._jobs_table.setItem(row, window._JOB_LABEL_COLUMN, label_item)
    for row in range(len(jobs)):
        selection_model.select(
            window._jobs_table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    return jobs


def test_main_window_dashboard_delete_key_summarises_then_applies_each_action(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_delete_key.mufinder", "DashboardDeleteKey")
    running_job, failed_job, completed_job = _dashboard_with_jobs(
        window, service, doc, [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED]
    )

    events: list[object] = []
    monkeypatch.setattr(
        window,
        "_prompt_dashboard_bulk_action",
        lambda **counts: events.append(("prompt", counts)) or True,
    )
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: events.append(("cancel", jobs)))
    monkeypatch.setattr(
        window,
        "_reset_failed_jobs",
        lambda jobs, cleanup_choice=None: events.append(("reset", jobs, cleanup_choice)),
    )
    monkeypatch.setattr(
        window,
        "_delete_dashboard_jobs",
        lambda jobs, confirmed=False: events.append(("delete", jobs, confirmed)),
    )

    window._on_dashboard_delete_key_requested()

    # One prompt, before anything is touched, then each action already confirmed.
    assert events == [
        (
            "prompt",
            {"cancel_count": 1, "reset_count": 1, "delete_count": 1, "skipped_imported_count": 0},
        ),
        ("cancel", [running_job]),
        ("reset", [failed_job], True),
        ("delete", [completed_job], True),
    ]
    window.close()


def test_main_window_dashboard_delete_key_cancelled_summary_touches_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_delete_none.mufinder", "DashboardDeleteNone")
    jobs = _dashboard_with_jobs(
        window, service, doc, [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED]
    )

    touched: list[str] = []
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: None)
    monkeypatch.setattr(window, "_cancel_jobs", lambda *_a, **_k: touched.append("cancel"))
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda *_a, **_k: touched.append("reset"))
    monkeypatch.setattr(window, "_delete_dashboard_jobs", lambda *_a, **_k: touched.append("delete"))

    window._on_dashboard_delete_key_requested()

    assert touched == []
    assert all(job.id in doc.jobs for job in jobs)
    window.close()


@pytest.mark.parametrize(
    ("counts", "expected_lines", "has_cleanup_box"),
    [
        (
            {"cancel_count": 2, "reset_count": 1, "delete_count": 3},
            [
                "\u2022 Cancel 2 running jobs",
                "\u2022 Reset 1 failed job to ready",
                "\u2022 Delete 3 finished job records and linked results",
            ],
            True,
        ),
        (
            {"cancel_count": 0, "reset_count": 0, "delete_count": 1},
            ["\u2022 Delete 1 finished job record and linked results"],
            False,
        ),
    ],
)
def test_main_window_dashboard_bulk_prompt_lists_only_what_applies(
    qapp, monkeypatch: pytest.MonkeyPatch, counts, expected_lines, has_cleanup_box
) -> None:
    window = MainWindow()
    seen: dict[str, object] = {}

    def _fake_exec(box):
        seen["text"] = box.text()
        seen["lines"] = box.informativeText().splitlines()
        seen["checkbox"] = box.checkBox()
        seen["default"] = box.defaultButton().text()
        if box.checkBox() is not None:
            box.checkBox().setChecked(True)
        apply_button = next(button for button in box.buttons() if button.text() == "Apply")
        monkeypatch.setattr(main_window_module.QMessageBox, "clickedButton", lambda _self: apply_button)
        return 0

    monkeypatch.setattr(main_window_module.QMessageBox, "exec", _fake_exec)

    result = window._prompt_dashboard_bulk_action(**counts)

    total = sum(counts.values())
    assert seen["text"] == f"Apply to {total} selected job{'' if total == 1 else 's'}?"
    assert seen["lines"] == expected_lines
    assert (seen["checkbox"] is not None) is has_cleanup_box
    assert seen["default"] != "Apply"
    assert result is has_cleanup_box
    window.close()


def test_main_window_dashboard_bulk_prompt_cancel_returns_none(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()

    def _fake_exec(box):
        cancel_button = next(button for button in box.buttons() if button.text() != "Apply")
        monkeypatch.setattr(main_window_module.QMessageBox, "clickedButton", lambda _self: cancel_button)
        return 0

    monkeypatch.setattr(main_window_module.QMessageBox, "exec", _fake_exec)

    assert window._prompt_dashboard_bulk_action(cancel_count=1, reset_count=1, delete_count=0) is None
    window.close()


def test_main_window_dashboard_backspace_shortcut_uses_delete_handler(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_backspace_key.mufinder", "DashboardBackspaceKey")
    running_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING
    window._document = doc
    window._refresh_jobs_table()
    window._jobs_table.selectRow(0)
    cancelled: list[list[Job]] = []
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: False)

    assert window._jobs_backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    window._jobs_backspace_shortcut.activated.emit()

    assert cancelled == [[running_job]]
    window.close()


def test_main_window_dashboard_delete_key_confirms_before_deleting_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_delete_confirm.mufinder", "DashboardDeleteConfirm")
    (job,) = _dashboard_with_jobs(window, service, doc, [JobStatus.DEFINED])

    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: None)
    window._on_dashboard_delete_key_requested()
    assert job.id in doc.jobs

    # The cancelled prompt left the row and its selection in place.
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: False)
    window._on_dashboard_delete_key_requested()
    assert job.id not in doc.jobs
    window.close()


def test_main_window_reset_failed_jobs_can_clean_output_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_failed_cleanup.mufinder", "ResetFailedCleanup")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / "stale.err").write_text("old failure", encoding="utf-8")

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: True)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert sorted(path.name for path in output_dir.iterdir()) == [f"{job.label}.cell", f"{job.label}.param"]
    assert doc.provenance[-1].action == "job_reset"
    window.close()


def test_main_window_reset_failed_jobs_no_failed_shows_info(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_no_failed.mufinder", "ResetNoFailed")
    job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._reset_failed_jobs([job])

    assert infos == ["No resettable jobs"]
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


def test_main_window_reset_failed_jobs_accepts_cancelled_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cancelled_job.mufinder", "ResetCancelledJob")
    job = service.add_job(
        doc,
        label="job_cancelled",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.CANCELLED
    job.error_message = "stopped"
    window._document = doc

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert job.error_message is None
    window.close()


def test_main_window_reset_failed_jobs_cancelled_by_user(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cancelled.mufinder", "ResetCancelled")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    window._document = doc

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: None)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    window.close()


def test_main_window_reset_failed_jobs_warns_when_cleanup_fails(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cleanup_warn.mufinder", "ResetCleanupWarn")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stale.log").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: True)
    monkeypatch.setattr(window, "_clear_directory_contents", lambda _path: (_ for _ in ()).throw(OSError("denied")))

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert warnings == ["Directory cleanup issues"]
    window.close()


def test_main_window_inspect_failed_jobs_no_failed_shows_info(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_no_failed.mufinder", "InspectNoFailed")
    job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._inspect_failed_jobs([job])

    assert infos == ["No failed jobs"]
    window.close()


def test_main_window_failure_diagnostics_and_inspect_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_failed.mufinder", "InspectFailed")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "process failed"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    err_path = output_dir / f"{job.label}.err"
    err_path.write_text("line 1\nfatal error\nline 3\n", encoding="utf-8")

    diagnostics = window._failure_diagnostics_for_job(job)
    assert "Recorded failure message" in diagnostics
    assert "2: fatal error" in diagnostics

    captured_details: list[str] = []

    class _FakeMessageBox:
        class Icon:
            Information = 1

        def __init__(self, *_args, **_kwargs) -> None:
            self._details = ""

        def setIcon(self, *_args, **_kwargs) -> None:
            return None

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            return None

        def setText(self, *_args, **_kwargs) -> None:
            return None

        def setInformativeText(self, *_args, **_kwargs) -> None:
            return None

        def setDetailedText(self, details: str) -> None:
            self._details = details

        def setSizeGripEnabled(self, *_args, **_kwargs) -> None:
            return None

        def setWindowFlag(self, *_args, **_kwargs) -> None:
            return None

        def exec(self) -> int:
            captured_details.append(self._details)
            return 0

    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    window._inspect_failed_jobs([job])

    assert len(captured_details) == 1
    assert "job_failed.err" in captured_details[0]
    assert "2: fatal error" in captured_details[0]
    window.close()


def test_main_window_failure_diagnostics_prefers_numbered_error_files(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_numbered_err.mufinder", "InspectNumberedErr")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job_failed.0001.err").write_text("rank 1 error\n", encoding="utf-8")
    (output_dir / "job_failed.0002.err").write_text("rank 2 error\n", encoding="utf-8")
    (output_dir / "unrelated.err").write_text("other error\n", encoding="utf-8")

    candidates = window._error_file_candidates(job, output_dir)
    diagnostics = window._failure_diagnostics_for_job(job)

    assert [path.name for path in candidates] == ["job_failed.0001.err", "job_failed.0002.err"]
    assert "[job_failed.0001.err]" in diagnostics
    assert "[job_failed.0002.err]" in diagnostics
    assert "unrelated.err" not in diagnostics
    window.close()


def test_main_window_failure_diagnostics_includes_slurm_output(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_slurm_out.mufinder", "InspectSlurmOut")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.remote_job_id = "771300"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm-771300.out").write_text(
        "running job\nslurm error: out of memory\n",
        encoding="utf-8",
    )

    candidates = window._error_file_candidates(job, output_dir)
    diagnostics = window._failure_diagnostics_for_job(job)

    assert [path.name for path in candidates] == ["slurm-771300.out"]
    assert "[slurm-771300.out]" in diagnostics
    assert "2: slurm error: out of memory" in diagnostics
    window.close()


def test_main_window_failure_diagnostics_without_output_directory(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_no_output.mufinder", "InspectNoOutput")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    diagnostics = window._failure_diagnostics_for_job(job)

    assert "Output directory: not found" in diagnostics
    window.close()


def test_main_window_error_file_candidates_fallback_and_excerpt_helpers(
    qapp, tmp_path: Path
) -> None:
    output_dir = tmp_path / "errors"
    output_dir.mkdir()
    (output_dir / "job_x.err").write_text("line 1\nerror here\nline 3\n", encoding="utf-8")
    (output_dir / "empty.err").write_text("", encoding="utf-8")

    service = ProjectService()
    doc = service.create_project(tmp_path / "helpers_doc.mufinder", "HelpersDoc")
    job = service.add_job(
        doc,
        label="job_x",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    window = MainWindow()
    candidates = window._error_file_candidates(job, output_dir)
    assert [path.name for path in candidates] == ["job_x.err"]

    assert MainWindow._read_error_contexts(output_dir / "empty.err") == "[File is empty]"
    assert MainWindow._read_error_contexts(output_dir / "missing.err").startswith("Could not read file")
    assert "2: error here" in MainWindow._read_error_contexts(output_dir / "job_x.err")
    window.close()


def test_read_bounded_tail_small_file_matches_read_text(tmp_path: Path) -> None:
    path = tmp_path / "small.out"
    content = "first line\nsecond line\nthird line\n"
    path.write_text(content, encoding="utf-8")

    assert MainWindow._read_bounded_tail(path) == content
    assert MainWindow._read_error_contexts(path) != "[File is empty]"
    assert MainWindow._read_text_excerpt(path) == content.strip()


def test_read_bounded_tail_large_file_returns_only_tail(tmp_path: Path) -> None:
    path = tmp_path / "huge.out"
    limit = MainWindow._TAIL_READ_LIMIT_BYTES

    first_line = "FIRST-LINE-SENTINEL " + ("x" * 200) + "\n"
    filler_line = "y" * 500 + "\n"
    filler_count = (limit // len(filler_line)) + 10
    last_line = "LAST-LINE-SENTINEL\n"

    with path.open("w", encoding="utf-8") as handle:
        handle.write(first_line)
        for _ in range(filler_count):
            handle.write(filler_line)
        handle.write(last_line)

    assert path.stat().st_size > limit

    tail_text = MainWindow._read_bounded_tail(path)
    assert len(tail_text.encode("utf-8")) <= limit + len("[... truncated] ...\n")
    assert "FIRST-LINE-SENTINEL" not in tail_text
    assert "LAST-LINE-SENTINEL" in tail_text
    assert tail_text.startswith("[... truncated] ...\n")

    excerpt = MainWindow._read_text_excerpt(path, max_chars=10**9)
    assert "FIRST-LINE-SENTINEL" not in excerpt
    assert "LAST-LINE-SENTINEL" in excerpt

    error_contexts = MainWindow._read_error_contexts(path)
    assert "FIRST-LINE-SENTINEL" not in error_contexts


def test_main_window_prompt_failed_job_cleanup_choices(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeMessageBox:
        class Icon:
            Question = 1

        class StandardButton:
            Yes = 1
            No = 2
            Cancel = 3

        _answer = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def setIcon(self, *_args, **_kwargs) -> None:
            return None

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            return None

        def setText(self, *_args, **_kwargs) -> None:
            return None

        def setInformativeText(self, *_args, **_kwargs) -> None:
            return None

        def setStandardButtons(self, *_args, **_kwargs) -> None:
            return None

        def setDefaultButton(self, *_args, **_kwargs) -> None:
            return None

        def exec(self) -> int:
            return _FakeMessageBox._answer

    window = MainWindow()
    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_failed_job_cleanup(2) is True

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.No
    assert window._prompt_failed_job_cleanup(2) is False

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Cancel
    assert window._prompt_failed_job_cleanup(2) is None
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

    window._on_inspect_failed_from_convergence([300.0])
    window._on_reset_failed_from_convergence([300.0])

    assert inspected == [[job]]
    assert reset == [[job]]
    window.close()


def test_main_window_close_invokes_runner_shutdown(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    called: list[bool] = []
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: called.append(True))

    window.close()

    assert called == [True]


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
    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._STATUS_COLUMN).text() == "ready"
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
    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._STATUS_COLUMN).text() == "ready"
    assert window._convergence_view._cutoff_table.item(1, window._convergence_view._STATUS_COLUMN).text() == "ready"
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

    assert window._convergence_view._cutoff_table.rowCount() == 2
    assert window._convergence_view._cutoff_table.item(0, window._convergence_view._CUTOFF_COLUMN).text() == "300"
    assert window._convergence_view._cutoff_table.item(1, window._convergence_view._CUTOFF_COLUMN).text() == "360"
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
    assert window._convergence_view._cutoff_table.rowCount() == 0
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
    assert window._convergence_view._cutoff_table.rowCount() == 0
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
    assert window._convergence_view._cutoff_table.rowCount() == 1
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

    window._remove_cutoff_from_definition(definition, 300.0)

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


def test_main_window_delete_path_retries_transient_permission_error(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    target_dir = tmp_path / "transient_dir"
    target_dir.mkdir()
    (target_dir / "inner.txt").write_text("x", encoding="utf-8")

    real_rmtree = main_window_module.shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Access is denied", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(main_window_module.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(main_window_module.time, "sleep", lambda _delay: None)

    window._delete_path(target_dir)

    assert calls["n"] >= 2
    assert not target_dir.exists()
    window.close()


def test_main_window_error_file_candidates_handles_missing_directory(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "missing_errors.mufinder", "MissingErrors")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    assert window._error_file_candidates(job, tmp_path / "does_not_exist") == []
    window.close()


def test_main_window_submit_jobs_locally_handles_no_eligible_defined_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_no_defined.mufinder", "SubmitNoDefined")
    job = service.add_job(
        doc,
        label="job_done",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.COMPLETED
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._submit_jobs_locally([job])

    assert infos == ["No eligible jobs"]
    window.close()


def test_main_window_submit_jobs_locally_dialog_rejected(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_dialog_reject.mufinder", "SubmitDialogReject")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    (doc.project_root / "files" / "convergence_cutoff" / job.label).mkdir(parents=True, exist_ok=True)
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
            return 0

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs))

    window._submit_jobs_locally([job])

    assert submit_calls == []
    assert job.status == JobStatus.DEFINED
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
        300.0: main_window_module.CutoffRunState(status="completed", total_energy=-10.0, max_force=0.2),
        400.0: main_window_module.CutoffRunState(status="completed", total_energy=-10.02, max_force=0.15),
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

    _select_route(window, "convergence_cutoff")
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
    assert window._kpoint_convergence_view._table.item(0, window._kpoint_convergence_view._GRID_COLUMN).text() == "3 x 3 x 2"
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
            parent=None,
        ) -> None:
            captured["cutoff_data"] = cutoff_data
            captured["num_atoms"] = num_atoms
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

    _select_route(window, "convergence_kpoint")
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

    _select_route(reopened, "convergence")

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

    _select_route(window, "convergence")

    assert (
        window._convergence_overview_view._cutoff_card._plot._placeholder.text()
        == "No adopted cutoff analysis is pinned yet.\nOpen the analysis dialog in the child workflow and adopt a project default to pin it here."
    )
    assert window._convergence_overview_view._cutoff_card._plot._canvas.isHidden() is True
    window.close()


# ---------------------------------------------------------------------------
# Job-lifecycle hardening: retry retrieval (rec #4) + dismiss (rec #8)
# ---------------------------------------------------------------------------
def _connect_cluster_runner(window, cluster_name: str) -> None:
    """Put the ClusterController into a connected state for the given cluster."""

    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = cluster_name


def _make_failed_cluster_job(service, doc, *, label: str, cluster: str = "scarf") -> Job:
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.execution_target = cluster
    job.remote_job_id = "99123"
    job.remote_directory = "/remote/work/" + label
    return job


def test_dashboard_retry_retrieval_button_enablement(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry.mufinder", "Retry")

    connected_job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")
    disconnected_job = _make_failed_cluster_job(service, doc, label="disc", cluster="other")
    no_meta_job = _make_failed_cluster_job(service, doc, label="nometa", cluster="scarf")
    no_meta_job.remote_directory = None
    local_job = service.add_job(
        doc,
        label="localjob",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    local_job.status = JobStatus.FAILED
    local_job.execution_target = "local"

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    window._refresh_jobs_table()

    def retry_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN)
        return None

    assert retry_btn_for(connected_job.id).text() == "Retry retrieval"
    assert retry_btn_for(connected_job.id).isEnabled() is True
    # Cluster not connected for this job's target -> disabled.
    assert retry_btn_for(disconnected_job.id).isEnabled() is False
    # Missing remote directory -> disabled.
    assert retry_btn_for(no_meta_job.id).isEnabled() is False
    # Local job -> disabled.
    assert retry_btn_for(local_job.id).isEnabled() is False
    window.close()


def test_dashboard_retry_retrieval_click_dispatches_to_runner(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry_click.mufinder", "RetryClick")
    job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: tmp_path)

    retrieve_calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        window._cluster_runner,
        "retrieve_outputs_for",
        lambda job_id, **kwargs: retrieve_calls.append((job_id, kwargs)),
    )
    monkeypatch.setattr(window, "_request_dashboard_flush", lambda **kwargs: None)

    window._refresh_jobs_table()
    row = 0
    retry_btn = window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN)
    retry_btn.click()

    # The job moved to RETRIEVING and the runner was asked to re-download.
    assert job.status == JobStatus.RETRIEVING
    assert len(retrieve_calls) == 1
    called_job_id, kwargs = retrieve_calls[0]
    assert called_job_id == job.id
    assert kwargs["remote_job_id"] == "99123"
    assert kwargs["remote_directory"] == "/remote/work/conn"
    window.close()


def test_dashboard_dismiss_hides_row_after_flush(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss.mufinder", "Dismiss")
    failed_job = service.add_job(
        doc,
        label="failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_job.status = JobStatus.FAILED
    running_job = service.add_job(
        doc,
        label="running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING

    window._document = doc
    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 2

    # The running job's Dismiss button must be disabled; the failed job's enabled.
    def dismiss_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_DISMISS_COLUMN)
        return None

    assert dismiss_btn_for(failed_job.id).isEnabled() is True
    assert dismiss_btn_for(running_job.id).isEnabled() is False

    dismiss_btn_for(failed_job.id).click()
    # Dismissal schedules a coalesced flush; force it to rebuild the table now.
    window._flush_dashboard()

    # After dismissal + rebuild, the failed row disappears; running row remains.
    assert failed_job.dismissed_at is not None
    assert window._jobs_table.rowCount() == 1
    remaining = window._jobs_table.item(0, window._JOB_LABEL_COLUMN)
    assert remaining.data(Qt.ItemDataRole.UserRole) == str(running_job.id)
    # Provenance milestone recorded.
    assert doc.provenance[-1].action == "job_dismissed"
    window.close()


def test_dashboard_dismiss_multi_select(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_multi.mufinder", "DismissMulti")
    job0 = service.add_job(doc, label="d0", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job1 = service.add_job(doc, label="d1", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job0.status = JobStatus.FAILED
    job1.status = JobStatus.STOPPED

    window._document = doc
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    dismiss_btn = window._jobs_table.cellWidget(0, window._JOB_DISMISS_COLUMN)
    assert isinstance(dismiss_btn, SelectionSnapshotPushButton)
    dismiss_btn.selection_snapshot_rows = (0, 1)
    dismiss_btn.clicked.emit()
    window._flush_dashboard()

    assert job0.dismissed_at is not None
    assert job1.dismissed_at is not None
    assert window._jobs_table.rowCount() == 0
    window.close()


def _add_dashboard_job(service, doc, label, status):
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = status
    return job


def _dashboard_row_widgets(window, row):
    """Snapshot the cell widgets + items for a dashboard row (identity checks)."""
    return {
        "label": window._jobs_table.item(row, window._JOB_LABEL_COLUMN),
        "status": window._jobs_table.item(row, window._JOB_STATUS_COLUMN),
        "resubmit": window._jobs_table.cellWidget(row, window._JOB_SUBMIT_COLUMN),
        "retry": window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN),
        "cancel": window._jobs_table.cellWidget(row, window._JOB_CANCEL_COLUMN),
        "inspect": window._jobs_table.cellWidget(row, window._JOB_INSPECT_COLUMN),
        "reset": window._jobs_table.cellWidget(row, window._JOB_RESET_COLUMN),
        "dismiss": window._jobs_table.cellWidget(row, window._JOB_DISMISS_COLUMN),
    }


def test_dashboard_reconcile_reuses_widgets_when_jobs_unchanged(
    qapp, tmp_path: Path
) -> None:
    """Two consecutive refreshes with the same jobs must reuse every cell widget."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_reuse.mufinder", "ReconcileReuse")
    _add_dashboard_job(service, doc, "r0", JobStatus.RUNNING)
    _add_dashboard_job(service, doc, "r1", JobStatus.FAILED)
    window._document = doc

    window._refresh_jobs_table()
    before = [_dashboard_row_widgets(window, row) for row in range(window._jobs_table.rowCount())]
    assert len(before) == 2

    window._refresh_jobs_table()
    after = [_dashboard_row_widgets(window, row) for row in range(window._jobs_table.rowCount())]
    assert len(after) == 2

    # Every item and every button is the identical object after re-refresh.
    for row_before, row_after in zip(before, after, strict=True):
        for key in row_before:
            assert row_before[key] is row_after[key], f"row widget {key} was recreated"
    window.close()


def test_dashboard_reconcile_status_change_updates_in_place(
    qapp, tmp_path: Path
) -> None:
    """A status change updates the status cell + button enablement without widget churn."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_status.mufinder", "ReconcileStatus")
    job = _add_dashboard_job(service, doc, "s0", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    before = _dashboard_row_widgets(window, 0)
    assert before["status"].text() == JobStatus.RUNNING.value
    # Running: cancel enabled, resubmit/inspect/reset disabled.
    assert before["cancel"].isEnabled() is True
    assert before["resubmit"].isEnabled() is False
    assert before["inspect"].isEnabled() is False
    assert before["reset"].isEnabled() is False

    job.status = JobStatus.FAILED
    window._refresh_jobs_table()
    after = _dashboard_row_widgets(window, 0)

    # Same widget objects, updated text + enablement.
    for key in before:
        assert before[key] is after[key], f"row widget {key} was recreated on status change"
    assert after["status"].text() == JobStatus.FAILED.value
    assert after["cancel"].isEnabled() is False
    assert after["resubmit"].isEnabled() is True
    assert after["inspect"].isEnabled() is True
    assert after["reset"].isEnabled() is True
    window.close()


def test_dashboard_reconcile_remove_job_removes_only_its_row(
    qapp, tmp_path: Path
) -> None:
    """Dismissing a job removes exactly its row; the surviving row keeps its widgets."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_remove.mufinder", "ReconcileRemove")
    job0 = _add_dashboard_job(service, doc, "x0", JobStatus.FAILED)
    _add_dashboard_job(service, doc, "x1", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 2
    # Capture the surviving (running) row's widgets before removal.
    survivor_before = _dashboard_row_widgets(window, 1)

    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()

    assert window._jobs_table.rowCount() == 1
    remaining_id = window._jobs_table.item(0, window._JOB_LABEL_COLUMN).data(Qt.ItemDataRole.UserRole)
    assert remaining_id != str(job0.id)
    # The survivor's widgets are preserved (moved up to row 0 identity intact).
    survivor_after = _dashboard_row_widgets(window, 0)
    for key in survivor_before:
        assert survivor_before[key] is survivor_after[key], f"survivor widget {key} was recreated"
    window.close()


def test_dashboard_reconcile_add_job_inserts_row_preserving_others(
    qapp, tmp_path: Path
) -> None:
    """Adding a job appends a row; existing rows keep their widgets."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_add.mufinder", "ReconcileAdd")
    _add_dashboard_job(service, doc, "a0", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 1
    first_before = _dashboard_row_widgets(window, 0)

    new_job = _add_dashboard_job(service, doc, "a1", JobStatus.FAILED)
    window._refresh_jobs_table()

    assert window._jobs_table.rowCount() == 2
    first_after = _dashboard_row_widgets(window, 0)
    for key in first_before:
        assert first_before[key] is first_after[key], f"existing widget {key} was recreated on insert"
    # New job landed in the appended row (document iteration order).
    assert window._jobs_table.item(1, window._JOB_LABEL_COLUMN).data(Qt.ItemDataRole.UserRole) == str(new_job.id)
    window.close()


def test_dashboard_reconcile_button_dispatches_correct_job_id_after_churn(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After several reconciliations, a button still dispatches its own job id."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_closure.mufinder", "ReconcileClosure")
    job0 = _add_dashboard_job(service, doc, "c0", JobStatus.FAILED)
    job1 = _add_dashboard_job(service, doc, "c1", JobStatus.FAILED)
    window._document = doc

    resubmitted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids)))

    window._refresh_jobs_table()
    # Churn: remove job0 (dismiss), refresh, re-add nothing but refresh again.
    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    job0.dismissed_at = None
    window._refresh_jobs_table()  # job0 comes back (fresh row/closure)

    # Locate job1's resubmit button by id and click it — must dispatch job1.
    def resubmit_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_SUBMIT_COLUMN)
        return None

    resubmit_btn_for(job1.id).click()
    resubmit_btn_for(job0.id).click()

    assert resubmitted == [[job1.id], [job0.id]]
    window.close()


def utc_now_for_test():
    from datetime import datetime

    return datetime.now(UTC)


def test_main_window_round_trips_mem_per_cpu_in_last_submission_config(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mem_per_cpu.mufinder", "ClusterMem")
    window._document = doc

    window._persist_last_cluster_submission_config(
        ClusterSubmissionConfig(
            cluster_name="SCARF",
            working_directory="/work4/isis/alice/project_z",
            username="alice",
            job_name_prefix="positivemuon",
            num_processes=144,
            walltime="12:00:00",
            mem_per_cpu_gb=6.0,
        ),
        workflow_type="muon_relaxation/positivemuon",
    )

    persisted = doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY]
    assert persisted["mem_per_cpu_gb"] == 6.0

    restored = window._load_last_cluster_submission_config(
        workflow_type="muon_relaxation/positivemuon"
    )
    assert restored is not None
    assert restored.mem_per_cpu_gb == 6.0
    window.close()


@pytest.mark.parametrize("stored", [None, "", "bad-memory", 0, -2.0])
def test_main_window_loads_absent_or_invalid_mem_per_cpu_as_unset(
    qapp, tmp_path: Path, stored: object
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mem_per_cpu_bad.mufinder", "ClusterMem")
    window._document = doc
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "SCARF",
        "num_processes": 144,
        "mem_per_cpu_gb": stored,
    }

    restored = window._load_last_cluster_submission_config()

    assert restored is not None
    assert restored.mem_per_cpu_gb is None
    window.close()


# --------------------------------------------------------------------------- #
# Dismissed failed jobs stay resubmittable from their workflow row
# --------------------------------------------------------------------------- #
#
# Regression for the stranded-row bug: dismissing a failed job from the jobs
# dashboard removes its Resubmit/Reset controls, but its workflow row kept
# projecting "failed" and stayed unsubmittable forever. The row now projects
# "ready" and its submit resets the *same* job back to defined before dispatch.


def _fail_and_dismiss(window: MainWindow, service: ProjectService, job: Job) -> None:
    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._on_dismiss_from_dashboard([job.id])
    assert job.dismissed_at is not None
    assert job.status is JobStatus.FAILED


def test_dismissed_pristine_relaxation_job_row_is_resubmittable(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_pristine.mufinder", "DismissedPristine")
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
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [{"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []}],
    }
    window._refresh_ui_after_project_change()

    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._pristine_relaxation_view._STATUS_COLUMN
    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "failed"

    # Dismissing from the dashboard removes the job's only generic controls...
    window._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    # ...so the workflow row becomes the recovery path.
    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "ready"

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert job.dismissed_at is None
    assert job.error_message is None
    # The row reused its bound job — no duplicate was minted for it.
    assert len(doc.jobs) == 1
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    window.close()


def test_dismissed_convergence_cutoff_job_row_is_resubmittable(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_cutoff.mufinder", "DismissedCutoff")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
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
    window._refresh_ui_after_project_change()

    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._convergence_view._STATUS_COLUMN
    assert window._convergence_view._cutoff_table.item(0, status_column).text() == "failed"

    window._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    assert window._convergence_view._cutoff_table.item(0, status_column).text() == "ready"
    # The cutoff panel's Submit actions are gated on the projected row status,
    # so they become available again for the dismissed row.
    assert not workflow_table_utils.workflow_row_is_locked(
        window._convergence_view._row_status(0)
    )

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._convergence_controller.submit([300.0], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert job.dismissed_at is None
    assert len(doc.jobs) == 1
    # The lazy reset must NOT detach the sweep point from its job the way the
    # explicit "Reset failed jobs" action does, or the row would go back to
    # "ready" with no job and a second submit would mint a duplicate.
    assert definition.parameters["job_ids_by_cutoff"] == {
        main_window_module.MainWindow._cutoff_key(300.0): str(job.id)
    }
    window.close()


def test_dismissed_row_submit_aborts_when_cleanup_prompt_cancelled(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_cancel.mufinder", "DismissedCancel")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
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
    _fail_and_dismiss(window, service, job)

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: None)

    window._convergence_controller.submit([300.0], target="local")

    assert submitted == []
    assert job.status is JobStatus.FAILED
    assert job.dismissed_at is not None
    window.close()


def test_non_dismissed_failed_row_keeps_failed_status_and_is_not_reset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed job that was never dismissed keeps its current behavior.

    Its recovery path is still the dashboard Resubmit/Reset controls, which are
    still on screen because the row was never filtered out.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "undismissed_cutoff.mufinder", "UndismissedCutoff")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
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
    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()

    assert (
        window._convergence_view._cutoff_table.item(
            0, window._convergence_view._STATUS_COLUMN
        ).text()
        == "failed"
    )

    prompted: list[int] = []
    monkeypatch.setattr(
        window, "_prompt_failed_job_cleanup", lambda count: prompted.append(count) or False
    )
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args, **_kwargs: infos.append((args[1], args[2])),
    )

    window._convergence_controller.submit([300.0], target="local")

    assert prompted == []
    assert job.status is JobStatus.FAILED
    assert job.dismissed_at is None
    window.close()


def test_dismissed_row_submit_regenerates_job_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resubmitted job's inputs are rebuilt from the current configuration.

    ``materialize`` only rewrites a job directory for a DEFINED job, and it runs
    while the plan is being built — before the lazy reset. Without a second
    materialize pass the row would resubmit against the stale ``.cell`` the
    failed run left behind (``_clear_directory_contents`` deliberately preserves
    ``.cell``/``.param``), diverging from the explicit reset-then-submit path.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_rewrite.mufinder", "DismissedRewrite")
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
    job = window._pristine_relaxation_controller.materialize(["row_1"])[0]
    cell_path = doc.project_root / "files" / "pristine_relaxation" / job.label / f"{job.label}.cell"
    assert "KPOINT_MP_GRID" not in cell_path.read_text(encoding="utf-8")

    _fail_and_dismiss(window, service, job)

    # The configuration changes while the row is stranded.
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert "KPOINT_MP_GRID 4 4 4" in cell_path.read_text(encoding="utf-8")
    assert len(doc.jobs) == 1
    window.close()


# ---------------------------------------------------------------------------
# Per-row "Reset Failed Jobs..." coverage for the workflows that lacked it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("handler_name", "controller_attr", "keys"),
    [
        ("_on_reset_failed_from_pristine_relaxation", "_pristine_relaxation_controller", ["relax_1"]),
        ("_on_reset_failed_from_muon_relaxation", "_muon_relaxation_controller", ["positivemuon:siteA"]),
        ("_on_reset_failed_from_energy_barriers", "_energy_barriers_controller", ["path-a"]),
        ("_on_reset_failed_from_uep", "_uep_controller", ["uep_1"]),
        ("_on_reset_failed_from_kpoint_convergence", "_kpoint_convergence_controller", [(2, 2, 2)]),
    ],
)
def test_main_window_row_reset_handlers_delegate_to_reset_failed_jobs(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
    controller_attr: str,
    keys: list[object],
) -> None:
    window = MainWindow()
    job = Job(
        label="row-job",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_relaxation",
        status=JobStatus.FAILED,
    )

    requested_keys: list[list[object]] = []
    reset: list[list[Job]] = []
    monkeypatch.setattr(
        getattr(window, controller_attr),
        "jobs_for_keys",
        lambda values: (requested_keys.append(list(values)), [job])[1],
    )
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda jobs: reset.append(jobs))

    getattr(window, handler_name)(keys)

    assert requested_keys == [keys]
    assert reset == [[job]]
    window.close()


@pytest.mark.parametrize(
    ("view_attr", "handler_name", "payload"),
    [
        ("_pristine_relaxation_view", "_on_reset_failed_from_pristine_relaxation", ["relax_1"]),
        ("_muon_relaxation_view", "_on_reset_failed_from_muon_relaxation", ["positivemuon:siteA"]),
        ("_uep_view", "_on_reset_failed_from_uep", ["uep_1"]),
        ("_kpoint_convergence_view", "_on_reset_failed_from_kpoint_convergence", [(2, 2, 2)]),
        ("_convergence_view", "_on_reset_failed_from_convergence", [300.0]),
    ],
)
def test_main_window_connects_workflow_reset_signals(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    view_attr: str,
    handler_name: str,
    payload: list[object],
) -> None:
    # The reset signal must actually reach MainWindow: k-point and UEP shipped a
    # declared-and-connected signal that no widget ever emitted, which is exactly
    # the failure this asserts against.
    window = MainWindow()
    received: list[object] = []
    monkeypatch.setattr(window, handler_name, received.append)

    getattr(window, view_attr).reset_failed_requested.emit(payload)

    assert received == [payload]
    window.close()


def test_main_window_connects_energy_barriers_reset_signal(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The energy-barriers view is created lazily, so its wiring lives in
    # _ensure_energy_barriers_view rather than __init__.
    window = MainWindow()
    view = window._ensure_energy_barriers_view()
    received: list[object] = []
    monkeypatch.setattr(window, "_on_reset_failed_from_energy_barriers", received.append)

    view.reset_failed_requested.emit(["path-a"])

    assert received == [["path-a"]]
    window.close()


def test_main_window_reset_refreshes_the_lazily_created_workflow_views(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Energy barriers and quantum corrections are not repainted by
    # _refresh_jobs_table, so a row reset from either table would leave a stale
    # "failed" status behind without these two refreshes.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_refresh.mufinder", "ResetRefresh")
    job = service.add_job(
        doc,
        label="neb-job",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_neb",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    refreshed: list[str] = []
    monkeypatch.setattr(
        window, "_refresh_energy_barriers_view", lambda: refreshed.append("energy_barriers")
    )
    monkeypatch.setattr(
        window, "_refresh_quantum_corrections_view", lambda: refreshed.append("quantum")
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._reset_failed_jobs([job])

    assert refreshed == ["energy_barriers", "quantum"]
    assert job.status == JobStatus.DEFINED
    window.close()


def _stage_quantum_row_with_singlepoint(
    window: MainWindow,
    service: ProjectService,
    doc,
    *,
    singlepoint_status: JobStatus,
):
    """Stage one Quantum Corrections row whose stage-1 singlepoint has finished.

    Mirrors the scaffold
    ``test_main_window_reset_quantum_corrections_to_stage_one_clears_shared_job_state``
    builds, but leaves the singlepoint status configurable so the reset gate can
    be exercised for a completed *and* a failed stage 1.
    """

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

    singlepoint_job = service.add_job(
        doc,
        label="quantum_singlepoint_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
    )
    singlepoint_job.status = singlepoint_status
    definition.parameters["singlepoint_job_ids_by_label"] = {str(site.id): str(singlepoint_job.id)}

    candidate = window._quantum_source_candidate_by_key(str(site.id))
    assert candidate is not None
    return definition, candidate, singlepoint_job


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
    definition, candidate, _ = _stage_quantum_row_with_singlepoint(
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
    definition, candidate, _ = _stage_quantum_row_with_singlepoint(
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
    definition, candidate, singlepoint_job = _stage_quantum_row_with_singlepoint(
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


def test_main_window_row_reset_clears_the_displayed_status_for_a_bound_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-row reset must repaint the row, not just mutate the Job object.

    Cutoff convergence clears its row by detaching the job from
    ``job_ids_by_cutoff``; the other six workflows keep their label->job-id
    mapping pointing at the reset job, so the row only clears because the job is
    now ``defined`` and ``effective_row_status`` projects a submittable status.
    This drives the real refresh path (no stubbed refresh) and asserts the Status
    *cell*, which is what the user actually sees.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "row_reset_repaint.mufinder", "RowResetRepaint")
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
    failed_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    failed_job.status = JobStatus.FAILED
    definition.parameters["job_ids_by_label"] = {"row_1": str(failed_job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [{"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []}],
    }

    window._refresh_ui_after_project_change()

    view = window._pristine_relaxation_view
    assert view._row_status(0) == "failed"
    # The context menu offers the reset precisely because the row reads "failed".
    assert view._row_status(0) in workflow_table_utils.RESETTABLE_ROW_STATUSES

    # Decline the output cleanup so this asserts the state change alone.
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._on_reset_failed_from_pristine_relaxation(["row_1"])

    assert failed_job.status == JobStatus.DEFINED
    # The row stays bound to the same job - no duplicate is minted - and now
    # reads as submittable again.
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(failed_job.id)}
    assert view._row_status(0) not in workflow_table_utils.RESETTABLE_ROW_STATUSES
    assert view._row_status(0) in READY_WORKFLOW_ROW_STATUSES
    window.close()


# --------------------------------------------------------------------------- #
# "Continue relaxation…" — WP6 controller action wiring
# --------------------------------------------------------------------------- #


class _FakeContinueRelaxationDialog:
    """Test double for ``ContinueRelaxationDialog``: records constructor kwargs
    and returns a canned accept/cancel result without any Qt event loop."""

    captured_kwargs: list[dict[str, object]] = []
    accept: bool = True
    result_to_return: tuple[dict[str, object], tuple[str, ...]] = ({}, ())

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        type(self).captured_kwargs.append(kwargs)

    def exec(self) -> int:
        return (
            main_window_module.QDialog.DialogCode.Accepted
            if type(self).accept
            else main_window_module.QDialog.DialogCode.Rejected
        )

    def edited_parameters(self) -> tuple[dict[str, object], tuple[str, ...]]:
        return type(self).result_to_return


@pytest.fixture()
def fake_continue_dialog(monkeypatch: pytest.MonkeyPatch):
    _FakeContinueRelaxationDialog.captured_kwargs = []
    _FakeContinueRelaxationDialog.accept = True
    _FakeContinueRelaxationDialog.result_to_return = ({}, ())
    monkeypatch.setattr(
        main_window_module, "ContinueRelaxationDialog", _FakeContinueRelaxationDialog
    )
    return _FakeContinueRelaxationDialog


class _FakeClusterSubmissionDialogForContinue:
    DialogCode = SimpleNamespace(Accepted=1)

    def __init__(self, **_kwargs) -> None:
        pass

    def exec(self) -> int:
        return 1

    def config(self):
        return SimpleNamespace(
            cluster_name="alpha",
            working_directory="/remote/work",
            username="alice",
            job_name_prefix="continue",
            num_processes=16,
            walltime="01:00:00",
        )

    def password(self) -> str:
        return ""


def _wire_fake_cluster_submission(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, *, cluster_name: str = "alpha"
) -> list[dict[str, object]]:
    """Mirrors ``test_main_window_resubmits_failed_cluster_job_...``'s cluster
    doubles: connect + submit succeed, and any existence-check/overwrite-delete
    call fails the test outright, since a continuation must never reach them."""

    cluster = ClusterConfig(
        name=cluster_name,
        hostname=f"{cluster_name}.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    monkeypatch.setattr(
        main_window_module, "ClusterSubmissionDialog", _FakeClusterSubmissionDialogForContinue
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda _config, password=None: None
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("Continue must never check for an existing remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("Continue must never delete the preserved remote directory"),
    )
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new-remote-job-id",
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: None)
    return submit_calls


def _pristine_continue_fixture(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "continue_pristine.mufinder", "ContinuePristine")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={"geom_max_iter": 120})
    output_root = doc.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=output_root,
    )
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": False, "params": {"geom_max_iter": 120}, "added_advanced": []}
        ],
    }
    window._refresh_ui_after_project_change()
    materialized = window._pristine_relaxation_controller.materialize(["row_1"])
    assert materialized == [job]
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    return window, service, doc, definition, job, output_root, param_path


def _muon_continue_fixture(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "continue_muon.mufinder", "ContinueMuon")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.1, 0.2, 0.3], status="staged")
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="positive_muon",
        parameter_values={"geom_max_iter": 120},
    )
    output_root = doc.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_muon_relaxation_config(
            main_window_module.MuonRelaxationConfig(rows=(row,))
        ),
        output_directory=output_root,
    )
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = window._serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(row,))
    )
    window._refresh_ui_after_project_change()
    materialized = window._muon_relaxation_controller.materialize(["positivemuon:Site 1"])
    assert materialized == [job]
    seed_name = definition.parameters["job_seed_names_by_label"]["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed_name / f"{seed_name}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    return window, service, doc, definition, job, output_root, param_path


def test_continue_relaxation_pristine_cluster_dismissed_failed_row_end_to_end(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )

    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    job.remote_job_id = "old-remote-id"
    job.error_message = "previous castep failure"
    job.dismissed_at = datetime.now(UTC)
    window._refresh_ui_after_project_change()
    assert window._pristine_relaxation_view._row_status(0) == "ready"  # dismissed -> displays "ready"

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    cleanup_prompted: list[int] = []
    monkeypatch.setattr(
        window,
        "_prompt_failed_job_cleanup",
        lambda count: cleanup_prompted.append(count) or pytest.fail(
            "Continue must never trigger the dismissed-job cleanup prompt"
        ),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # Row params mutated in place.
    config = window._deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    assert {row.label: row.parameter_values for row in config.rows} == {
        "row_1": {"geom_max_iter": 500}
    }
    # Same directory rewritten with the new parameters.
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered
    # No duplicate job; job_ids_by_label untouched.
    assert list(doc.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    # Submitted into the same remote directory without any existence/overwrite prompt.
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/pristine_relax_row_1"
    assert submit_calls[0]["allow_existing_remote_directory"] is True
    assert submit_calls[0]["reuse_existing_files"] is False
    # dismissed_at/error/old remote id cleared by prepare_job_continuation, and
    # the job is back in an active state after a successful cluster submit.
    assert job.status == JobStatus.QUEUED
    assert job.dismissed_at is None
    assert job.error_message is None
    assert job.remote_job_id == "new-remote-job-id"
    assert job.remote_directory == "/remote/work/pristine_relax_row_1"
    assert cleanup_prompted == []
    # Journal-visible provenance milestone.
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_muon_cluster_completed_row_end_to_end(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )

    job.execution_target = "alpha"
    job.status = JobStatus.COMPLETED
    job.remote_directory = "/remote/work/positivemuon_site1"
    window._refresh_ui_after_project_change()

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 777}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")

    window._on_continue_relaxation_requested_from_muon_relaxation("positivemuon:Site 1")

    config = window._deserialize_muon_relaxation_config(definition.parameters["configuration"])
    assert {row.charge_state: row.parameter_values for row in config.rows} == {
        "positive_muon": {"geom_max_iter": 777}
    }
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 777" in rendered
    assert list(doc.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"positivemuon:Site 1": str(job.id)}
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/positivemuon_site1"
    assert submit_calls[0]["allow_existing_remote_directory"] is True
    assert job.status == JobStatus.QUEUED
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_local_job_round_trip(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    assert job.execution_target == "local"

    # A local job's checkpoint probe is a plain Path check: write a real
    # ``.check`` file so the probe reports "found".
    checkpoint_path = param_path.parent / f"{job.label}.check"
    checkpoint_path.write_text("", encoding="utf-8")

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 42}, ())
    local_submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: local_submitted.append(list(jobs))
    )
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda *_a, **_k: pytest.fail("A local job must submit through the local path"),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    probe = fake_continue_dialog.captured_kwargs[-1]["checkpoint_probe"]
    assert probe.status == main_window_module.CheckpointProbeResult.found().status
    assert [[j.id for j in jobs] for jobs in local_submitted] == [[job.id]]
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 42" in rendered
    window.close()


def test_continue_relaxation_cluster_not_connected_dialog_still_opens_with_unavailable_probe(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    assert window._cluster_runner.is_connected is False

    monkeypatch.setattr(
        window._cluster_runner,
        "check_remote_files",
        lambda *_a, **_k: pytest.fail("Not connected: the checkpoint probe must not hit the network"),
    )
    fake_continue_dialog.accept = False  # cancel: no submission wiring needed

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert len(fake_continue_dialog.captured_kwargs) == 1
    probe = fake_continue_dialog.captured_kwargs[0]["checkpoint_probe"]
    assert probe.status == main_window_module.CheckpointProbeResult.unavailable("x").status
    assert "alpha" in probe.reason
    window.close()


def test_continue_relaxation_cancel_leaves_everything_unchanged(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    before_config = copy.deepcopy(definition.parameters["configuration"])

    fake_continue_dialog.accept = False
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda *_a, **_k: pytest.fail("Cancel must not submit")
    )
    monkeypatch.setattr(
        window, "_submit_jobs_on_cluster", lambda *_a, **_k: pytest.fail("Cancel must not submit")
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert definition.parameters["configuration"] == before_config
    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    assert not any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_gating_disabled_running_enabled_not_converged(
    qapp, tmp_path: Path
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )

    job.status = JobStatus.RUNNING
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is False

    job.status = JobStatus.QUEUED
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is False

    # A completed-but-unconverged job displays "not converged" on the row, but
    # its raw status (COMPLETED) is continuable.
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=False)
    doc.runs[run.id] = run
    window._refresh_ui_after_project_change()
    assert window._pristine_relaxation_view._row_status(0) == "not converged"
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is True
    window.close()


def test_continue_relaxation_unedited_real_dialog_shows_no_parameter_changes(
    qapp, tmp_path: Path
) -> None:
    """WP4 contract check: ``old_param_text`` must be rendered through the same
    ``render_param_text`` callable applied to the row's pre-edit values -- the
    exact values seeded into the dialog's own editor -- so an unedited dialog
    reads "No parameter changes." This constructs the REAL
    ``ContinueRelaxationDialog`` (not the fake double the other tests use) to
    prove the seeding path this module wires up actually satisfies that
    contract, rather than merely asserting the fake was called correctly."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED

    seed = window._pristine_relaxation_view.continue_relaxation_dialog_seed("row_1")
    assert seed is not None

    def render_param_text(values, advanced):
        return window._pristine_relaxation_controller.render_row_param_text(
            "row_1", values, advanced
        )

    resolved_values, resolved_advanced = window._resolve_continue_relaxation_seed_values(seed)
    old_param_text = render_param_text(resolved_values, resolved_advanced)

    dialog = main_window_module.ContinueRelaxationDialog(
        **seed,
        old_param_text=old_param_text,
        render_param_text=render_param_text,
        checkpoint_probe=main_window_module.CheckpointProbeResult.absent(),
        execution_target=job.execution_target,
        reused_directory=str(param_path.parent),
        parent=window,
    )
    try:
        assert dialog._diff_table.rowCount() == 0
        assert dialog._diff_no_changes_label.isHidden() is False
        assert dialog._diff_table.isHidden() is True
    finally:
        dialog.close()
    window.close()


def test_continue_relaxation_muon_unedited_real_dialog_shows_no_parameter_changes(
    qapp, tmp_path: Path
) -> None:
    """Muon counterpart of the pristine no-changes contract check above,
    using the charge-qualified key end to end (finding 7)."""

    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED

    seed = window._muon_relaxation_view.continue_relaxation_dialog_seed("positivemuon:Site 1")
    assert seed is not None

    def render_param_text(values, advanced):
        return window._muon_relaxation_controller.render_row_param_text(
            "positivemuon:Site 1", values, advanced
        )

    resolved_values, resolved_advanced = window._resolve_continue_relaxation_seed_values(seed)
    old_param_text = render_param_text(resolved_values, resolved_advanced)

    dialog = main_window_module.ContinueRelaxationDialog(
        **seed,
        old_param_text=old_param_text,
        render_param_text=render_param_text,
        checkpoint_probe=main_window_module.CheckpointProbeResult.absent(),
        execution_target=job.execution_target,
        reused_directory=str(param_path.parent),
        parent=window,
    )
    try:
        assert dialog._diff_table.rowCount() == 0
        assert dialog._diff_no_changes_label.isHidden() is False
        assert dialog._diff_table.isHidden() is True
    finally:
        dialog.close()
    window.close()


# --------------------------------------------------------------------------- #
# Opus audit follow-ups (findings 1-4)
# --------------------------------------------------------------------------- #


def test_continue_relaxation_warns_and_does_not_submit_when_materialize_fails(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 1 (MainWindow half): when continue_job returns None after
    the user accepted the dialog -- e.g. because materialize swallowed an
    expansion error and rewrote nothing -- the row must not be silently
    submitted with stale on-disk inputs, and the user must be told."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())

    # Force the continuation's own materialize call to fail, exactly like the
    # real one does on a caught ValueError/StructureBridgeError.
    window._pristine_relaxation_controller.materialize = lambda keys: []

    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda *_a, **_k: pytest.fail("must not submit")
    )
    monkeypatch.setattr(
        window, "_submit_jobs_on_cluster", lambda *_a, **_k: pytest.fail("must not submit")
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert len(warnings) == 1
    # The job was still transitioned (prepare_job_continuation ran before the
    # failed materialize) -- it stays recoverable -- but nothing was submitted,
    # and no provenance milestone claims a continuation that never happened.
    assert job.status == JobStatus.DEFINED
    assert "GEOM_MAX_ITER : 500" not in param_path.read_text(encoding="utf-8")
    assert not any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


class _FakeClusterSubmissionDialogSelectsCluster:
    """Like ``_FakeClusterSubmissionDialogForContinue``, but the user picks a
    DIFFERENT cluster than the one the row was preselected onto -- proving the
    cross-cluster misroute guard (audit finding 2)."""

    DialogCode = SimpleNamespace(Accepted=1)
    selected_cluster_name = "beta"

    def __init__(self, **_kwargs) -> None:
        pass

    def exec(self) -> int:
        return 1

    def config(self):
        return SimpleNamespace(
            cluster_name=type(self).selected_cluster_name,
            working_directory="/remote/work",
            username="bob",
            job_name_prefix="continue",
            num_processes=16,
            walltime="01:00:00",
        )

    def password(self) -> str:
        return ""


def test_continue_relaxation_cross_cluster_misroute_guard(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 2: the submission dialog preselects the job's original
    cluster, but the user can still pick a different one. The continuation
    branch in ``_submit_jobs_on_cluster`` must not fire in that case -- a
    continued job's ``remote_directory`` string means nothing on a different
    cluster's filesystem. This asserts the NORMAL path is taken instead: the
    remote-directory existence check runs (it is skipped entirely for a
    same-cluster continuation), and nothing is deleted."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())

    alpha = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    beta = ClusterConfig(
        name="beta",
        hostname="beta.example",
        username="bob",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    monkeypatch.setattr(
        main_window_module, "ClusterSubmissionDialog", _FakeClusterSubmissionDialogSelectsCluster
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [alpha, beta])
    monkeypatch.setattr(
        main_window_module,
        "load_cluster_config",
        lambda _settings, name: beta if name == "beta" else alpha,
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda _config, password=None: None
    )
    checked_paths: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: checked_paths.append(path) or False,
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("must never delete on the misrouted path"),
    )
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new-remote-job-id",
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: None)

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # The continuation branch was NOT taken: the existence check ran, and the
    # job was submitted with allow_existing_remote_directory=False /
    # reuse_existing_files=False -- the ordinary cold-target path, not a
    # blind directory reuse onto a cluster that never saw this job.
    assert len(checked_paths) == 1
    assert len(submit_calls) == 1
    assert submit_calls[0]["allow_existing_remote_directory"] is False
    assert submit_calls[0]["reuse_existing_files"] is False
    assert job.execution_target == "beta"
    window.close()


def test_continue_relaxation_pristine_context_menu_action_invokes_the_real_handler(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 3: prove the MainWindow wiring lines themselves (the
    ``set_continuable_predicate``/``continue_relaxation_requested.connect``
    calls in ``__init__``) are live, by driving the action through the REAL
    row context menu rather than calling the handler method directly."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    # COMPLETED (not FAILED/STOPPED/CANCELLED): continuable but NOT resettable,
    # so the shared FakeMenu's "auto-fire the one enabled action" behavior
    # (tests/support/reset_menu.py) cannot land on "Reset Failed Jobs..."
    # instead of the action under test.
    job.status = JobStatus.COMPLETED
    window._refresh_ui_after_project_change()
    fake_continue_dialog.accept = False  # the handler running is all this proves

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(window._pristine_relaxation_view._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True

    action.triggered.emit()

    assert len(fake_continue_dialog.captured_kwargs) == 1
    window.close()


def test_continue_relaxation_muon_context_menu_action_invokes_the_real_handler(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Muon counterpart of the pristine context-menu wiring check above."""

    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.COMPLETED
    window._refresh_ui_after_project_change()
    fake_continue_dialog.accept = False

    install_fake_menu(monkeypatch)
    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    menu = open_reset_menu(table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True

    action.triggered.emit()

    assert len(fake_continue_dialog.captured_kwargs) == 1
    window.close()


def test_await_remote_file_check_reports_found(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit finding 4: drive the real nested-QEventLoop wait in
    ``_await_remote_file_check`` end to end (worker thread + signal), rather
    than only unit-testing ``ClusterController.check_remote_files`` in
    isolation (already covered by ``tests/app/test_cluster_controller.py``)."""

    window = MainWindow()
    job_id = uuid4()
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_file_exists",
        lambda path: path == "/remote/x/seed.check",
    )

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result == {"existing": ["/remote/x/seed.check"], "errors": {}}
    window.close()


def test_await_remote_file_check_reports_absent(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    job_id = uuid4()
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", lambda _path: False)

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result == {"existing": [], "errors": {}}
    window.close()


def test_await_remote_file_check_reports_worker_exception_as_unavailable(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    job_id = uuid4()

    def _boom(_path: str) -> bool:
        raise ConnectionError("stat failed")

    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", _boom)

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result["existing"] == []
    assert result["errors"] == {job_id: "stat failed"}

    # And the probe helper built on top of it maps that into an unavailable
    # CheckpointProbeResult rather than raising or blocking the dialog.
    job = Job(
        label="probe_job",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
        execution_target="alpha",
        remote_directory="/remote/x",
    )
    job.status = JobStatus.FAILED
    monkeypatch.setattr(window, "_is_cluster_connected_for_job", lambda _job: True)
    monkeypatch.setattr(window, "_job_seed_name", lambda _job: "seed")
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", _boom)

    probe = window._continue_relaxation_checkpoint_probe(job)

    from mufinder.services.cluster_runner import CheckpointProbeStatus

    assert probe.status == CheckpointProbeStatus.UNAVAILABLE
    assert "stat failed" in probe.reason
    window.close()


def test_main_window_dashboard_scope_prompt_cancel_does_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_scope_cancel.mufinder", "SnapScopeCancel")
    for label in ("j0", "j1"):
        job = service.add_job(
            doc,
            label=label,
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, title, body, *a: asked.append((title, body)) or SelectionScope.CANCEL,
    )
    window._refresh_jobs_table()

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    cancel_btn.selection_snapshot_rows = (0, 1)
    cancel_btn.clicked.emit()

    assert asked == [("Apply to selected jobs?", "Cancel all 2 selected jobs?")]
    assert cancelled == []
    window.close()


@pytest.mark.parametrize(
    ("kind", "status", "column_attr", "handler", "expected_prompt"),
    [
        (
            "cancel",
            JobStatus.RUNNING,
            "_JOB_CANCEL_COLUMN",
            "_on_cancel_from_dashboard",
            ("Cancel job", "Cancel job 'j0'?", "", "Cancel job", "Keep running"),
        ),
        (
            "dismiss",
            JobStatus.FAILED,
            "_JOB_DISMISS_COLUMN",
            "_on_dismiss_from_dashboard",
            (
                "Dismiss job",
                "Dismiss job 'j0' from the dashboard?",
                "The job stays in the project; its workflow row can still resubmit it.",
                "Dismiss job",
            ),
        ),
    ],
)
def test_main_window_dashboard_single_row_destructive_button_is_confirmed(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind, status, column_attr, handler, expected_prompt
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / f"single_{kind}.mufinder", "Single")
    job = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = status
    window._document = doc
    acted: list[list[object]] = []
    monkeypatch.setattr(window, handler, lambda ids: acted.append(list(ids)))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    button = window._jobs_table.cellWidget(0, getattr(window, column_attr))
    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == []

    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == [[job.id]]
    assert prompts[0] == expected_prompt
    window.close()


def test_main_window_dashboard_multi_row_cancel_is_confirmed_by_the_scope_prompt_alone(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "multi_cancel.mufinder", "MultiCancel")
    jobs = []
    for label in ("j0", "j1"):
        job = service.add_job(
            doc,
            label=label,
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.RUNNING
        jobs.append(job)
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.ALL_SELECTED
    )
    window._refresh_jobs_table()

    button = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    button.selection_snapshot_rows = (0, 1)
    button.clicked.emit()

    assert [set(ids) for ids in cancelled] == [{job.id for job in jobs}]
    assert confirmed_destructive_actions == []
    window.close()


def test_main_window_reset_failed_jobs_uses_supplied_cleanup_choice_without_asking(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_preconfirmed.mufinder", "ResetPreconfirmed")
    job = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc
    asked: list[int] = []
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda count: asked.append(count))
    applied: list[tuple[list[Job], bool]] = []
    monkeypatch.setattr(
        window,
        "_apply_failed_job_reset",
        lambda jobs, *, cleanup_choice, detach_convergence: applied.append((list(jobs), cleanup_choice))
        or (0, []),
    )

    window._reset_failed_jobs([job], cleanup_choice=False)

    assert asked == []
    assert applied == [([job], False)]
    window.close()


def _dashboard_window_with_statuses(tmp_path: Path, name: str, statuses):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
    jobs = []
    for index, status in enumerate(statuses):
        job = service.add_job(
            doc,
            label=f"j{index}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = status
        jobs.append(job)
    window._document = doc
    window._refresh_jobs_table()
    return window, jobs


def _select_dashboard_rows(window, rows) -> None:
    selection_model = window._jobs_table.selectionModel()
    selection_model.clearSelection()
    for row in rows:
        selection_model.select(
            window._jobs_table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_main_window_dashboard_hint_appears_only_for_a_multi_row_selection(
    qapp, tmp_path: Path
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "hint", [JobStatus.FAILED, JobStatus.FAILED, JobStatus.RUNNING]
    )
    hint = window._jobs_selection_hint
    dock = window.findChild(main_window_module.QDockWidget, "jobDashboardDock")
    assert window._jobs_table.parent() is dock.widget()
    assert hint.parent() is dock.widget()
    assert hint.isHidden()

    _select_dashboard_rows(window, [0])
    assert hint.isHidden()

    _select_dashboard_rows(window, [0, 1])
    assert not hint.isHidden()
    assert hint.text() == "2 selected \u00b7 row buttons apply to all selected"
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    assert bool(window._jobs_table.cellWidget(0, window._JOB_RESET_COLUMN).property(prop)) is True
    assert bool(window._jobs_table.cellWidget(2, window._JOB_RESET_COLUMN).property(prop)) is False

    _select_dashboard_rows(window, [])
    assert hint.isHidden()
    window.close()


def _open_dashboard_menu(window, monkeypatch, row: int = 0):
    from tests.support.reset_menu import FakeMenu

    FakeMenu.instances = []
    monkeypatch.setattr(main_window_module, "QMenu", FakeMenu)
    rect = window._jobs_table.visualRect(window._jobs_table.model().index(row, 0))
    window._jobs_table.customContextMenuRequested.emit(rect.center())
    return FakeMenu.instances[-1]


def test_main_window_dashboard_menu_acts_on_the_eligible_selected_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed, cancelled_job) = _dashboard_window_with_statuses(
        tmp_path, "menu", [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED]
    )
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("resubmit", "_on_resubmit_from_dashboard"),
        ("inspect", "_on_inspect_failed_from_dashboard"),
        ("reset", "_on_reset_failed_from_dashboard"),
    ):
        monkeypatch.setattr(window, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    _select_dashboard_rows(window, [0, 1, 2])

    menu = _open_dashboard_menu(window, monkeypatch)

    # Entries that apply to only some of the three selected jobs say how many.
    assert [action.text for action in menu.actions] == [
        "Resubmit (2 of 3)",
        "Retry Retrieval",
        "Cancel (1 of 3)",
        "Inspect (1 of 3)",
        "Reset (2 of 3)",
        "Dismiss (2 of 3)",
        "Delete...",
    ]
    assert menu.separators_before == [6]
    assert {action.text for action in menu.actions if not action.enabled} == {"Retry Retrieval"}

    menu.action("Resubmit (2 of 3)").triggered.emit()
    menu.action("Inspect (1 of 3)").triggered.emit()
    menu.action("Reset (2 of 3)").triggered.emit()
    assert calls == [
        ("resubmit", [failed.id, cancelled_job.id]),
        ("inspect", [failed.id]),
        ("reset", [failed.id, cancelled_job.id]),
    ]
    window.close()


def test_main_window_dashboard_menu_confirms_cancel_and_dismiss(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed) = _dashboard_window_with_statuses(
        tmp_path, "menu_confirm", [JobStatus.RUNNING, JobStatus.FAILED]
    )
    calls: list[tuple[str, list[object]]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: calls.append(("cancel", list(ids))))
    monkeypatch.setattr(window, "_on_dismiss_from_dashboard", lambda ids: calls.append(("dismiss", list(ids))))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    _select_dashboard_rows(window, [0, 1])
    menu = _open_dashboard_menu(window, monkeypatch)

    menu.action("Cancel (1 of 2)").triggered.emit()
    assert calls == []
    menu.action("Cancel (1 of 2)").triggered.emit()
    menu.action("Dismiss (1 of 2)").triggered.emit()

    assert calls == [("cancel", [running.id]), ("dismiss", [failed.id])]
    assert prompts[0] == ("Cancel jobs", "Cancel 1 job?", "", "Cancel 1 job", "Keep running")
    assert prompts[2][1] == "Dismiss 1 job from the dashboard?"
    window.close()


def test_main_window_dashboard_menu_selects_the_clicked_row_and_offers_delete(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "menu_delete", [JobStatus.FAILED, JobStatus.FAILED, JobStatus.FAILED]
    )
    deleted: list[int] = []
    monkeypatch.setattr(window, "_on_dashboard_delete_key_requested", lambda: deleted.append(1))
    _select_dashboard_rows(window, [0, 1])

    menu = _open_dashboard_menu(window, monkeypatch, row=2)

    selected = sorted(index.row() for index in window._jobs_table.selectionModel().selectedRows())
    assert selected == [2]
    menu.action("Delete...").triggered.emit()
    assert deleted == [1]
    window.close()


@pytest.mark.parametrize(
    ("view_attr", "controller_attr"),
    [
        ("_pristine_relaxation_view", "_pristine_relaxation_controller"),
        ("_muon_relaxation_view", "_muon_relaxation_controller"),
        ("_quantum_corrections_view", "_quantum_corrections_controller"),
    ],
)
def test_main_window_connects_workflow_cancel_and_inspect_signals(
    qapp, monkeypatch: pytest.MonkeyPatch, view_attr, controller_attr
) -> None:
    window = MainWindow()
    controller = getattr(window, controller_attr)
    sentinel_jobs = [object()]
    monkeypatch.setattr(controller, "jobs_for_keys", lambda keys: sentinel_jobs if keys == ["k"] else [])
    cancelled: list[object] = []
    inspected: list[object] = []
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window, "_inspect_failed_jobs", lambda jobs: inspected.append(jobs))

    view = getattr(window, view_attr)
    view.cancel_requested.emit(["k"])
    view.inspect_failed_requested.emit(["k"])

    assert cancelled == [sentinel_jobs]
    assert inspected == [sentinel_jobs]
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
        view._add_cutoff(cutoff)
    sent: list[list[str]] = []
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, **_kwargs: sent.append(
            [job.label for job in jobs if job.status == JobStatus.DEFINED]
        ),
    )
    view._run_cluster_selected_cutoffs()
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

    assert view._cutoff_parameter_values == {}
    assert sorted(view.build_config().shared_params) == defaults
    status_column = view._STATUS_COLUMN
    assert [view._cutoff_table.item(row, status_column).text() for row in range(2)] == [
        "completed",
        "completed",
    ]

    view._add_cutoff(420.0)
    view._cutoff_table.selectAll()
    view._run_cluster_selected_cutoffs()

    # Only the new row is submitted; the completed rows keep their jobs.
    assert sent == [["cutoff_convergence_ecut_420"]]
    assert sorted((job.label, job.status) for job in doc.jobs.values()) == [
        ("cutoff_convergence_ecut_300", JobStatus.COMPLETED),
        ("cutoff_convergence_ecut_360", JobStatus.COMPLETED),
        ("cutoff_convergence_ecut_420", JobStatus.DEFINED),
    ]
    assert [view._cutoff_table.item(row, status_column).text() for row in range(2)] == [
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

    assert view._cutoff_parameter_values == {}
    view._cutoff_table.selectAll()
    view._run_cluster_selected_cutoffs()
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
