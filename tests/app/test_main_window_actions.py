"""Behavior tests for main-window command flows and error handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write

import mufinder.app.main_window as main_window_module
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.domain.models import ComparisonTable, MuonSite, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.project_service import ProjectService
from tests.support.qt import find_item_by_route
from tests.support.main_window import (
    select_route,
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


def test_supercell_editor_route_uses_project_page(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "supercell_route.mufinder", "SupercellRoute")
    window._document = doc
    window._populate_project_tree()

    window._set_active_route("supercell_editor")

    assert window._main_stack.currentWidget() == window._supercell_editor_page
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


def test_muon_charge_state_dialog_defaults_and_returns_both_states(qapp) -> None:
    dialog = main_window_module.MuonChargeStateDialog(site_count=2)

    assert dialog.objectName() == "muonChargeStateDialog"
    assert dialog.windowTitle() == "Create Muon Relaxation Jobs"
    assert dialog.selected_charge_states() == ("positive_muon",)

    dialog._both.setChecked(True)

    assert dialog.selected_charge_states() == ("positive_muon", "muonium")


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
