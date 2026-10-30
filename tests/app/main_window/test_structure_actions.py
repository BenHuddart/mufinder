"""Structure actions in the main window: host loading, supercell editor, comparisons and magnetic configuration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write

import mufinder.app.main_window as main_window_module
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.workflow_controllers.comparison_tables import comparison_route
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.project_service import ProjectService
from tests.support.main_window import select_route


def test_host_structure_route_switches_panel(main_window, tmp_path: Path) -> None:
    window = main_window
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
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
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
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "comparison_labels.mufinder", "ComparisonLabels"
    )

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

    monkeypatch.setattr(
        window._pristine_relaxation_controller, "completed_jobs", lambda: [completed]
    )
    monkeypatch.setattr(
        main_window_module,
        "run_comparison_job_dialog",
        lambda jobs, default_label, parent: ([completed.job_id], default_label),
    )

    window._pristine_relaxation_view.generate_comparison_requested.emit()

    tables = list(window._document.comparison_tables.values())
    assert len(tables) == 1
    assert tables[0].dft_labels == ["DFT (pristine_relax_alpha)"]

    window.close()


def test_comparison_table_title_edit_updates_project_tree_and_persists(
    main_window, tmp_path: Path
) -> None:
    window = main_window
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
    window._comparison_tables_controller.sync_view()
    window._populate_project_tree()
    window._set_active_route(comparison_route(table.id))

    widget = window._comparison_tables_view.table_widget(table.id)
    widget._title_edit.setText("  Renamed title  ")
    widget._title_edit.editingFinished.emit()

    assert doc.comparison_tables[table.id].label == "Renamed title"
    tree_item = window._find_tree_item_by_route(comparison_route(table.id))
    assert tree_item is not None
    assert tree_item.text(0) == "Renamed title"
    reloaded = service.load_project(doc.project_root)
    assert reloaded.comparison_tables[table.id].label == "Renamed title"
    window.close()


def test_host_magnetic_configuration_changes_update_document(main_window, tmp_path: Path) -> None:
    window = main_window
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
                "moments": [
                    {"label": "Fe1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"}
                ],
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
    assert reloaded.project.magnetic_structure["propagation_vector"] == pytest.approx(
        [0.0, 0.0, 0.25]
    )
    assert reloaded.project.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    window.close()


def test_load_structure_requested_handles_import_error(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "load_error_test.mufinder", "LoadErr")

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("bad.cif", "Structure Files"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warn")
    )

    def broken_import(*args, **kwargs):
        raise StructureLoadError("bad")

    monkeypatch.setattr(window._project_service, "import_structure_file", broken_import)

    window._on_load_structure_requested()
    assert warnings == ["warn"]
    window.close()


def test_supercell_editor_route_uses_project_page(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "supercell_route.mufinder", "SupercellRoute")
    window._document = doc
    window._populate_project_tree()

    window._set_active_route("supercell_editor")

    assert window._main_stack.currentWidget() == window._supercell_editor_page
    window.close()


@pytest.mark.parametrize(
    ("view_name", "handler_name", "missing_title", "status"),
    [
        (
            "_pristine_relaxation_view",
            "_on_pristine_relaxation_supercell_edit_requested",
            "No host structure",
            "Selected pristine-relaxation simulation cell",
        ),
        (
            "_muon_relaxation_view",
            "_on_muon_relaxation_supercell_edit_requested",
            "No host structure",
            "Selected muon-relaxation simulation cell",
        ),
        (
            "_uep_view",
            "_on_uep_supercell_edit_requested",
            "No source structure",
            "Selected UEP simulation cell",
        ),
    ],
)
def test_row_supercell_editor_guards_the_source_and_sets_the_chosen_matrix(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view_name: str,
    handler_name: str,
    missing_title: str,
    status: str,
) -> None:
    window = main_window
    view = getattr(window, view_name)
    handler = getattr(window, handler_name)
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )
    opened: list[dict[str, object]] = []
    chosen: list[list[list[int]] | None] = [[[2, 0, 0], [0, 2, 0], [0, 0, 1]]]
    cancelled = False

    class _FakeDialog:
        def selected_matrix(self):
            return chosen[0]

    def fake_open(**kwargs):
        opened.append(kwargs)
        return None if cancelled else _FakeDialog()

    monkeypatch.setattr(window, "_open_supercell_editor", fake_open)

    handler(0)
    assert warnings == [] and opened == []

    service = ProjectService()
    doc = service.create_project(tmp_path / "row_supercell.mufinder", "RowSupercell")
    window._document = doc
    row_id = 7

    handler(row_id)
    assert warnings == [missing_title]
    assert opened == []

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    expected_source = host
    if view_name == "_uep_view":
        expected_source = Structure(
            label="Relaxed host",
            lattice_vectors=[[5.1, 0.0, 0.0], [0.0, 5.1, 0.0], [0.0, 0.0, 5.1]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
        service.add_structure(doc, expected_source)
        doc.ui_state["uep_form"] = {"selected_source_structure_id": str(expected_source.id)}

    handler("not a row")
    assert opened == []

    cancelled = True
    handler(row_id)
    cancelled = False
    chosen[0] = None
    handler(row_id)
    assert view.row_supercell_matrix(row_id) is None
    assert window.statusBar().currentMessage() != status

    chosen[0] = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    handler(row_id)

    assert len(opened) == 3
    assert opened[-1]["source_structure"].id == expected_source.id
    assert opened[-1]["initial_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert opened[-1]["select_mode"] is True
    assert view.row_supercell_matrix(row_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert window.statusBar().currentMessage() == status

    handler(row_id)
    assert opened[-1]["initial_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    window.close()
