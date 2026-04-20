"""Behavior tests for main-window command flows and error handling."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write
from PySide6.QtWidgets import QTreeWidgetItem

import mufinder.app.main_window as main_window_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.cluster_setup import ClusterConfig
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.project_service import ProjectService


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
    root = window._project_tree.topLevelItem(0)
    assert root is not None
    item = _find_item_by_route(root, route)
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
    window._on_generate_requested("random", {"count": 2})

    assert warnings == ["warning"]
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

    root = window._project_tree.topLevelItem(0)
    host_item = root.child(0)
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


def test_convergence_simulation_cell_create_requested_can_reset_to_unit_cell(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def result_mode(self) -> str:
            return "unit_cell"

        def matrix(self) -> list[list[int]]:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    monkeypatch.setattr(main_window_module, "SupercellPreviewDialog", _FakeDialog)

    window._on_convergence_simulation_cell_create_requested()

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

    jobs = window._materialize_convergence_jobs([300.0])

    assert len(jobs) == 1
    cell_path = doc.project_root / "files" / "convergence_cutoff" / "conv_ecut_300" / "conv_ecut_300.cell"
    cell_text = cell_path.read_text(encoding="utf-8")
    assert "10 0 0" in cell_text
    assert cell_text.count("Si ") >= 2
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

    jobs = window._materialize_convergence_jobs([300.0])

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
    states = window._build_convergence_cutoff_states()

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


def test_main_window_build_states_uses_active_convergence_definition_only(qapp, tmp_path: Path) -> None:
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

    states = window._build_convergence_cutoff_states()

    assert old_definition.id != new_definition.id
    assert states[300.0].status == "ready"
    assert states[300.0].total_energy is None
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

    states = window._build_convergence_cutoff_states()

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
        window._cluster_runner,
        "connect_to_cluster",
        lambda config, password=None: connect_calls.append((config.name, password)),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(
        window._cluster_runner,
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
    assert initial_config.job_name_prefix == "convergence"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "pristine_relaxation"
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
        window._cluster_runner,
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
        window._cluster_runner,
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

    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner, "submit", lambda **_kwargs: "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "yes")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == ["/remote/work/conv_ecut_300"]
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

    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner, "submit", lambda **_kwargs: "99123")

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

    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", _connect)
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
    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh"))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("secret", True),
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

    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "remove_remote_directory", lambda path: deleted.append(path))
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner, "submit", lambda **kwargs: submitted.append(kwargs) or "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "no_to_all")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == []
    assert len(submitted) == 1
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

    monkeypatch.setattr(window._cluster_runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(window._cluster_runner, "submit", lambda **_kwargs: "99123")

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

    job.status = JobStatus.RUNNING
    run.status = JobStatus.RUNNING
    window._on_local_job_cancelled(job.id)
    assert job.status == JobStatus.CANCELLED
    assert run.status == JobStatus.CANCELLED
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

    monkeypatch.setattr(
        main_window_module,
        "parse_cell_file",
        lambda _path: SimpleNamespace(
            lattice_vectors=((5.1, 0.0, 0.0), (0.0, 5.2, 0.0), (0.0, 0.0, 5.3)),
            species=("Si",),
            fractional_positions=((0.0, 0.0, 0.0),),
        ),
    )
    monkeypatch.setattr(
        main_window_module,
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
    defined_job = service.add_job(
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
    submitted: list[list[object]] = []
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(window, "_on_submit_local_from_dashboard", lambda ids: submitted.append(list(ids)))
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids)))
    monkeypatch.setattr(window, "_on_inspect_failed_from_dashboard", lambda ids: inspected.append(list(ids)))
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: reset.append(list(ids)))

    window._refresh_jobs_table()

    assert window._jobs_table.contextMenuPolicy() == main_window_module.Qt.ContextMenuPolicy.NoContextMenu
    assert window._jobs_table.rowCount() == 2

    submit_button = window._jobs_table.cellWidget(0, window._JOB_SUBMIT_COLUMN)
    cancel_button = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    inspect_button = window._jobs_table.cellWidget(1, window._JOB_INSPECT_COLUMN)
    reset_button = window._jobs_table.cellWidget(1, window._JOB_RESET_COLUMN)

    assert isinstance(submit_button, main_window_module.QPushButton)
    assert isinstance(cancel_button, main_window_module.QPushButton)
    assert isinstance(inspect_button, main_window_module.QPushButton)
    assert isinstance(reset_button, main_window_module.QPushButton)

    assert submit_button.isEnabled() is False
    assert window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN).isEnabled() is True
    assert cancel_button.isEnabled() is True
    assert inspect_button.isEnabled() is True
    assert reset_button.isEnabled() is True

    cancel_button.click()
    inspect_button.click()
    reset_button.click()

    assert submitted == []
    assert cancelled == [[running_job.id]]
    assert inspected == [[failed_job.id]]
    assert reset == [[failed_job.id]]
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

        def exec(self) -> int:
            captured_details.append(self._details)
            return 0

    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    window._inspect_failed_jobs([job])

    assert len(captured_details) == 1
    assert "job_failed.err" in captured_details[0]
    assert "2: fatal error" in captured_details[0]
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

    window._on_submit_local_from_convergence([300.0])

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

    window._on_submit_local_from_convergence([300.0])

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

    window._on_submit_local_from_convergence([300.0, 360.0])

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
    assert window._build_convergence_cutoff_states()[300.0].status == "ready"
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
    monkeypatch.setattr(window, "_build_convergence_cutoff_states", lambda: states)

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


def test_main_window_convergence_examiner_state_changed_updates_ui_state(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_examiner_state_update.mufinder", "ConvExaminerStateUpdate")
    window._document = doc

    saves: list[bool] = []
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))

    window._on_convergence_examiner_state_changed({"results_text": "cached"})

    assert doc.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY] == {"results_text": "cached"}
    assert saves == [True]
    window.close()


def test_main_window_convergence_examiner_state_persists_across_project_reload(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    root = tmp_path / "conv_examiner_state_reload.mufinder"
    doc = service.create_project(root, "ConvExaminerStateReload")
    window._document = doc

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

    loaded = service.load_project(root)

    assert loaded.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY]["results_text"] == "Recommended cutoff: 500 eV"
    assert loaded.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY]["plot_data"]["cutoffs"] == [300.0, 400.0, 500.0]
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

    window._on_submit_local_from_kpoint_convergence([(2, 2, 2)])

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

    window._on_submit_local_from_kpoint_convergence([(2, 2, 2)])

    assert len(doc.jobs) == 1
    job = next(iter(doc.jobs.values()))
    assert job.status == JobStatus.SUBMITTING
    assert submitted and submitted[0]["seed_name"] == "kconv_kgrid_2x2x2"
    assert (doc.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.cell").exists()
    assert (doc.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param").exists()
    definition = next(iter(doc.calculations.values()))
    assert definition.parameters["job_ids_by_grid"][main_window_module.MainWindow._grid_key((2, 2, 2))] == str(job.id)
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

    window._on_submit_local_from_kpoint_convergence([(2, 2, 2), (2, 2, 2), (3, 3, 2)])

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
    monkeypatch.setattr(window, "_build_convergence_kpoint_states", lambda: states)

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, cutoff_data, *, num_atoms, sweep_kind, point_labels, initial_state=None, parent=None) -> None:
            captured["cutoff_data"] = cutoff_data
            captured["num_atoms"] = num_atoms
            captured["sweep_kind"] = sweep_kind
            captured["point_labels"] = point_labels
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
