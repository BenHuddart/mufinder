"""Structure actions in the main window: host loading, supercell editor, comparisons and magnetic configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QTableWidget
from ase import Atoms
from ase.io import write as ase_write

import mufinder.app.file_dialogs as file_dialogs_module
import mufinder.app.main_window as main_window_module
from mufinder.app.comparison_job_dialog import CompletedPristineJob
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.workflow_controllers.comparison_tables import comparison_route
from mufinder.domain.models import (
    ComparisonTable,
    MuonSite,
    Structure,
)
from mufinder.persistence.codec import to_jsonable
from mufinder.science.magnetic_structure import magnetic_model_for_host
from mufinder.services.magnetic_field_results import (
    CALCULATION_METHOD_VERSION,
    CALCULATION_VERSION_KEY,
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    dipolar_tensor_uses_older_method,
    field_result_uses_older_method,
    magnetic_field_result,
    owner_host_moments_moved,
    store_magnetic_field_results,
)
from mufinder.science.structure_io import StructureLoadError
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_fields import collinear_waves_model, spin_host_project
from tests.support.main_window import open_project_with_host, select_route
from tests.support.structures import add_relaxed_host, host_structure


def test_host_structure_route_switches_panel(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "route_test.mufinder", "Route")
    window._refresh_ui_after_project_change()

    select_route(window, "host_structure")

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
        file_dialogs_module.QFileDialog,
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


def test_a_relaxed_host_reads_by_its_relaxation_row_in_the_selector_and_facts(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(
        window, tmp_path, host=host_structure(source_file="imports/Si.cif")
    )
    relaxed = add_relaxed_host(document)
    window._refresh_host_structure_view()
    page = window._host_structure_view
    combo = page._active_host_combo
    facts = page._facts_card
    assert facts.value_label("Source").value() == "imports/Si.cif"

    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i) != combo.currentText())
    )

    assert document.project.host_structure_id == relaxed.id
    assert combo.currentText().startswith("Relaxed — relax_1 (")
    assert combo.toolTip() == combo.currentText()
    assert facts.value_label("Source").value() == "Pristine relaxation relax_1"
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
    window._refresh_navigator()
    window._set_active_route(comparison_route(table.id))

    widget = window._comparison_tables_view.table_widget(table.id)
    widget._title_edit.setText("  Renamed title  ")
    widget._title_edit.editingFinished.emit()

    assert doc.comparison_tables[table.id].label == "Renamed title"
    row = window._project_navigator.row(comparison_route(table.id))
    assert row is not None
    assert row.label == "Renamed title"
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
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        }
    )

    assert host.properties["atomic_spins"] == pytest.approx([2.0])
    assert host.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert window._document.project.magnetic_structure == {
        "propagation_vector": [0.0, 0.0, 0.25],
        "moments": [
            {
                "label": "Fe1",
                "atom_index": 0,
                "vector": [0.0, 0.0, 2.0],
                "coordinate_system": "cartesian",
            }
        ],
    }
    # A stale matrix in the payload no longer reaches the legacy project field.
    assert window._document.project.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    window._project_service.save_project(window._document)
    reloaded = window._project_service.load_project(window._document.project_root)
    reloaded_host = reloaded.structures[host.id]
    assert reloaded_host.properties["atomic_spins"] == pytest.approx([2.0])
    assert reloaded_host.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert reloaded.project.magnetic_structure is not None
    assert reloaded.project.magnetic_structure["propagation_vector"] == pytest.approx(
        [0.0, 0.0, 0.25]
    )
    assert reloaded.project.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    window.close()


def test_host_page_spin_edit_reaches_the_document(main_window, tmp_path: Path) -> None:
    window = main_window
    _, document, host = open_project_with_host(
        window, tmp_path, host=host_structure(species=["Fe"], positions=[(0.0, 0.0, 0.0)])
    )
    (atoms,) = [
        table for table in window.findChildren(QTableWidget) if table.accessibleName() == "Atoms"
    ]

    atoms.item(0, 4).setText("1.5")

    assert host.properties["atomic_spins"] == pytest.approx([1.5])
    assert host.properties["magnetic_structure"]["moments"][0]["vector"] == [0.0, 0.0, 1.5]
    assert document.project.magnetic_structure is host.properties["magnetic_structure"]
    window.close()


def test_host_page_spin_on_a_later_atom_puts_the_moment_on_that_atom(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    _, document, host = open_project_with_host(
        window, tmp_path, host=host_structure(species=["F", "Co"])
    )
    (atoms,) = [
        table for table in window.findChildren(QTableWidget) if table.accessibleName() == "Atoms"
    ]

    atoms.item(1, 4).setText("3")

    assert host.properties["atomic_spins"] == pytest.approx([0.0, 3.0])
    (component,) = magnetic_model_for_host(document.project.magnetic_structure, host).components
    assert component.label == "Co2"
    assert component.selector.values == (1,)
    window.close()


def test_a_spin_typed_on_a_moved_host_keeps_the_mark_on_its_magnetic_results(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    _service, document = spin_host_project(tmp_path)
    version = {CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION}
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [{"site_label": "site 1", "total_T": [0.0, 0.0, 1.0]}],
                **version,
            },
            "dipolar_tensor_selector": {"mode": "assigned"},
            "dipolar_tensor_results": [
                {"site_label": "site 1", "dipolar_tensor_T_per_muB": [[1.0] * 3] * 3, **version}
            ],
        },
    )
    window._document = document
    window._refresh_ui_after_project_change()
    (atoms,) = [
        table for table in window.findChildren(QTableWidget) if table.accessibleName() == "Atoms"
    ]

    atoms.item(1, 4).setText("2.5")

    assert not owner_host_moments_moved(document, MAGNETIC_FIELD_PAGE_OWNER)
    field = magnetic_field_result(document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE)
    tensors = magnetic_field_result(
        document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE
    )
    assert field_result_uses_older_method(field.data)
    assert dipolar_tensor_uses_older_method({**tensors.data["results"][0], **version})
    assert not window._magnetic_field_view._older_method_note.isHidden()
    window.close()


def _open_project_with_a_built_model(window, tmp_path: Path):
    """Save a one-atom Fe host whose project holds a builder model, then open it in ``window``."""

    built = to_jsonable(collinear_waves_model((0.0, 0.0, 0.5)).to_json())
    host = host_structure(
        species=["Fe"],
        positions=[(0.0, 0.0, 0.0)],
        properties={"magnetic_structure": copy.deepcopy(built)},
    )
    service = ProjectService()
    document = service.create_project(tmp_path / "built.mufinder", "Built")
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    document.project.magnetic_structure = copy.deepcopy(built)
    service.save_project(document)
    assert window.open_project(document.project_root)
    document = window._document
    return document, document.structures[host.id], built


def _assert_built_model_kept(window, document, host, built) -> None:
    assert document.project.magnetic_structure == built
    assert host.properties["magnetic_structure"] == built
    select_route(window, "magnetic_fields")
    shown = window._magnetic_field_view.collect_workspace_state()["magnetic_structure"]
    assert shown == built


def test_host_page_spin_edit_keeps_a_built_magnetic_model(main_window, tmp_path: Path) -> None:
    window = main_window
    document, host, built = _open_project_with_a_built_model(window, tmp_path)
    (atoms,) = [
        table for table in window.findChildren(QTableWidget) if table.accessibleName() == "Atoms"
    ]

    atoms.item(0, 4).setText("1.5")

    assert host.properties["atomic_spins"] == pytest.approx([1.5])
    _assert_built_model_kept(window, document, host, built)

    select_route(window, "host_structure")
    atoms.item(0, 4).setText("0")

    assert host.properties["atomic_spins"] == pytest.approx([0.0])
    _assert_built_model_kept(window, document, host, built)
    window.close()


def test_host_page_propagation_edit_keeps_a_built_magnetic_model(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    document, host, built = _open_project_with_a_built_model(window, tmp_path)
    (k_c,) = [
        edit
        for edit in window.findChildren(DecimalLineEdit)
        if edit.accessibleName() == "Propagation Vector c*"
    ]

    k_c.setValue(0.25)

    assert host.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    _assert_built_model_kept(window, document, host, built)
    window.close()


def test_load_structure_requested_handles_import_error(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "load_error_test.mufinder", "LoadErr")

    monkeypatch.setattr(
        file_dialogs_module.QFileDialog,
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


def test_host_page_button_opens_the_supercell_editor(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "supercell_button.mufinder", "SupercellButton")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc
    window._refresh_ui_after_project_change()
    opened: list[dict[str, object]] = []
    saved = window._project_service.create_workflow_supercell_asset(
        window._document,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    )

    class _FakeDialog:
        def matrix(self):
            return [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    def fake_open(**kwargs):
        opened.append(kwargs)
        return _FakeDialog()

    monkeypatch.setattr(window, "_open_supercell_editor", fake_open)

    window._host_structure_view._supercell_button.click()

    assert len(opened) == 1
    assert opened[0]["source_structure"].id == host.id
    assert opened[0]["select_mode"] is False
    assert opened[0]["workflow_stage"] is None
    assert opened[0]["initial_matrix"] == saved.properties["supercell_asset"]["matrix"]
    assert window._document.project.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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
            origin="pristine_relaxation",
            parent_structure_id=host.id,
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


def _choose_other_host(window) -> None:
    combo = window._host_structure_view._active_host_combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i) != combo.currentText())
    )


def _window_with_two_hosts(window, tmp_path: Path):
    _service, document, host = open_project_with_host(
        window, tmp_path, host=host_structure(label="CoF2")
    )
    # The relaxation that made the relaxed host was configured on the imported one.
    relaxed = add_relaxed_host(document, host=host)
    window._refresh_host_structure_view()
    return document, host, relaxed


def test_a_host_switch_asks_nothing_when_the_project_holds_no_work_for_the_host(
    main_window, tmp_path: Path, answered_choices
) -> None:
    window = main_window
    document, host, relaxed = _window_with_two_hosts(window, tmp_path)
    ProjectService().set_active_host_structure(document, relaxed.id)
    window._refresh_host_structure_view()

    # Nothing was configured on the relaxed host, so going back to the imported one asks nothing.
    _choose_other_host(window)

    assert answered_choices == []
    assert document.project.host_structure_id == host.id
    assert window._project_navigator.row("host_structure").detail == "Original — CoF2"
    window.close()


def test_a_host_switch_after_a_relaxation_asks_without_a_word_about_sites(
    main_window, tmp_path: Path, answered_choices
) -> None:
    window = main_window
    document, host, _relaxed = _window_with_two_hosts(window, tmp_path)

    _choose_other_host(window)

    (question,) = answered_choices
    assert question["detail"] == (
        "Every later stage will use the new host. Spins, the propagation vector and "
        "saved supercells are kept per host, so each host keeps its own."
    )
    assert document.project.host_structure_id == host.id
    window.close()


def test_a_host_switch_with_work_asks_first_and_declining_changes_nothing(
    main_window, tmp_path: Path, answered_choices
) -> None:
    window = main_window
    document, host, relaxed = _window_with_two_hosts(window, tmp_path)
    site = MuonSite(label="Mu 1", source_structure_id=host.id)
    document.sites[site.id] = site
    combo = window._host_structure_view._active_host_combo
    before = (combo.currentText(), len(document.provenance), dict(document.structures))

    _choose_other_host(window)

    new_label = next(
        entry.label
        for entry in ProjectService().list_promotable_host_structures(document)
        if entry.structure_id == relaxed.id
    )
    (question,) = answered_choices
    assert question["title"] == "Switch active host"
    assert question["text"] == f"Make '{new_label}' the active host?"
    assert question["choices"] == [("switch", "Switch host"), ("keep", "Cancel")]
    assert (question["default"], question["escape"]) == ("keep", "keep")
    assert question["detail"] == (
        "Every later stage will use the new host. Spins, the propagation vector and "
        "saved supercells are kept per host, so each host keeps its own. Sites generated "
        "for 'Original — CoF2' stay in the project and are marked as belonging to that host."
    )
    # The default answer keeps the host, and the selector is back on it.
    assert document.project.host_structure_id == host.id
    assert (combo.currentText(), len(document.provenance), dict(document.structures)) == before
    assert window._project_navigator.row("host_structure").detail == "Original — CoF2"
    window.close()


def test_a_host_switch_with_work_goes_ahead_when_accepted(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app import workflow_table_utils

    window = main_window
    document, host, relaxed = _window_with_two_hosts(window, tmp_path)
    ProjectService().create_workflow_supercell_asset(
        document,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    )
    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "switch")

    _choose_other_host(window)

    assert document.project.host_structure_id == relaxed.id
    assert window._host_structure_view._active_host_combo.currentText().startswith("Relaxed")
    assert window._project_navigator.row("host_structure").detail == "Relaxed — relax_1"
    window.close()
