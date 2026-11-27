"""Tests for the Host Structure page: layout, atom table, spins and selection sync.

The structure viewer's own behaviour is tested in ``test_atomic_structure_viewer.py``
(selection, camera, theme, shutdown) and ``test_atomic_structure_viewer_scene.py``
(meshes and overlays).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QSplitter

from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.workflow_table_utils import table_natural_width
from mufinder.services.host_structure_facts import structure_facts
from mufinder.services.project_service import ProjectService, PromotableHost
from tests.support.main_window import select_route
from tests.support.pyvista_fakes import count_created_interactors, install_fake_picker
from tests.support.qt import press_key, release_keyboard_modifiers
from tests.support.symmetry_hosts import cof2_host
from tests.support.structures import (
    hexagonal_magnesium_oxide_host,
    host_structure,
    magnetic_silicon_oxide_host,
)


def _assert_atom_table_fits(widget: HostStructureWidget) -> None:
    """Every column of the atom table is on screen and no horizontal scrollbar shows."""
    table = widget._atom_table
    last = table.columnCount() - 1
    assert not table.horizontalScrollBar().isVisible()
    assert table.columnViewportPosition(last) + table.columnWidth(last) <= table.viewport().width()
    # The column is as wide as its text needs, so nothing inside it is elided.
    assert table.columnWidth(last) >= table.horizontalHeader().sectionSizeHint(last)
    assert table.columnWidth(last) >= table.sizeHintForColumn(last)


@pytest.fixture
def designed_text_scale(qapp, monkeypatch):
    """The light theme at the designed text size, for tests of exact page widths.

    Another test may leave the session's application at a larger desktop text
    scale; the page's 620 px minimum holds at the designed size.
    """
    from mufinder.app import theme as theme_module

    monkeypatch.setattr(theme_module, "text_scale", lambda: 1.0)
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    yield qapp
    monkeypatch.undo()
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def _opening_width(widget: HostStructureWidget) -> int:
    """The pane's opening width: the table's or the one-line propagation row's, the wider."""
    return max(table_natural_width(widget._atom_table), widget._propagation_row_width())


def _crowded_host(atom_count: int):
    return host_structure(
        species=["Si"] * atom_count,
        positions=[(0.0, 0.0, index / atom_count) for index in range(atom_count)],
    )


def test_atom_table_fits_at_the_default_window_size(main_window, tmp_path: Path) -> None:
    """The window's default size shows the whole table, ``Spin (ħ/2)`` included."""
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "fit.mufinder", "Fit")
    window._refresh_ui_after_project_change()
    page = window._host_structure_view
    page.set_structure(magnetic_silicon_oxide_host())
    select_route(window, "host_structure")
    window.resize(1280, 864)
    window.show()
    QApplication.processEvents()

    _assert_atom_table_fits(page)
    assert page._viewer.width() >= page._viewer.minimumWidth()
    window.close()


def test_atom_table_fits_when_the_structure_loaded_before_the_page_was_shown(qapp) -> None:
    """A project opens with the page hidden; the split is still right on first show."""
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()

    _assert_atom_table_fits(widget)
    # The viewer, not the table, gets everything the table does not need.
    assert widget._viewer.width() > widget._atom_table.width()


def test_atom_table_fits_when_a_structure_loads_into_a_visible_page(qapp) -> None:
    widget = HostStructureWidget()
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()

    widget.set_structure(magnetic_silicon_oxide_host())
    qapp.processEvents()

    _assert_atom_table_fits(widget)


def test_atom_table_fits_with_four_digit_row_numbers_and_a_vertical_scrollbar(qapp) -> None:
    """At 1,000+ atoms the row numbers widen and the rows overflow; the split allows for both."""
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    narrow_gutter_width = table_natural_width(widget._atom_table)
    widget.set_structure(_crowded_host(1200))
    widget.resize(1000, 500)
    widget.show()
    qapp.processEvents()

    assert widget._atom_table.verticalScrollBar().isVisible()
    assert table_natural_width(widget._atom_table) > narrow_gutter_width
    _assert_atom_table_fits(widget)


def _host_with_a_wide_coordinate_after_row_1000():
    """1,200 atoms whose only long ``fX`` value, ``-0.8888``, is at row 1,101."""
    positions = [(0.1111, 0.0, index / 1200) for index in range(1200)]
    positions[1100] = (-0.8888, 0.0, 1100 / 1200)
    return host_structure(species=["Si"] * 1200, positions=positions)


def _assert_cell_shows_in_full(widget: HostStructureWidget, row: int, column: int) -> None:
    """The column is at least as wide as Qt would make it having measured every row."""
    table = widget._atom_table
    header = table.horizontalHeader()
    precision = header.resizeContentsPrecision()
    header.setResizeContentsPrecision(-1)
    try:
        assert table.columnWidth(column) >= table.sizeHintForColumn(column)
    finally:
        header.setResizeContentsPrecision(precision)
    assert (
        table.columnWidth(column)
        >= table.sizeHintForIndex(table.model().index(row, column)).width()
    )


def test_a_wider_value_first_appearing_after_row_1000_shows_in_full(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_host_with_a_wide_coordinate_after_row_1000())
    widget.resize(1000, 500)
    widget.show()
    qapp.processEvents()

    assert widget._atom_table.item(1100, 1).text() == "-0.8888"
    _assert_cell_shows_in_full(widget, 1100, 1)
    _assert_atom_table_fits(widget)


def test_a_wider_spin_typed_after_row_1000_shows_in_full(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_crowded_host(1200))
    widget.resize(1000, 500)
    widget.show()
    qapp.processEvents()

    widget._atom_table.item(1100, 4).setText("-123456789.5")
    qapp.processEvents()

    assert widget._atom_table.item(1100, 4).text() == "-123456789.5000"
    _assert_cell_shows_in_full(widget, 1100, 4)
    _assert_atom_table_fits(widget)


def test_columns_fit_a_small_host_loaded_after_a_large_one(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_crowded_host(1200))
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1000, 500)
    widget.show()
    qapp.processEvents()

    widget._atom_table.item(1, 4).setText("-123456789.5")
    qapp.processEvents()

    _assert_cell_shows_in_full(widget, 1, 4)


def test_viewer_keeps_its_minimum_and_the_table_scrolls_on_a_small_screen(
    designed_text_scale,
) -> None:
    qapp = designed_text_scale
    """Below the table's content width the table scrolls; the viewer keeps its 380 px.

    The table is wider than the propagation row above it, which holds the
    pane at its own width, so the pane is narrower than the table needs.
    """
    widget = HostStructureWidget()
    widget.set_structure(
        magnetic_silicon_oxide_host(
            species=("Si:labelled_species", "O"),
            properties={"atomic_spins": [-12345.5, 1.0]},
        )
    )
    widget.resize(620, 700)
    widget.show()
    qapp.processEvents()

    assert widget.width() == 620
    assert widget._viewer.minimumWidth() == 380
    assert widget._viewer.width() >= 380
    assert widget._atom_table.width() < table_natural_width(widget._atom_table)
    assert widget._atom_table.horizontalScrollBar().isVisible()


def test_window_growth_goes_to_the_viewer(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()
    table_width = widget._atom_table.width()

    widget.resize(1400, 700)
    qapp.processEvents()

    assert widget._atom_table.width() == table_width


def test_dragged_split_survives_a_spin_edit_and_resets_for_another_structure(qapp) -> None:
    widget = HostStructureWidget()
    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()

    splitter = widget._viewer.parentWidget()
    splitter.setSizes([500, 500])
    qapp.processEvents()
    dragged = widget._atom_table.width()
    assert dragged > table_natural_width(widget._atom_table)

    host.properties["atomic_spins"] = [2.0, -2.0]
    widget.set_structure(host)
    qapp.processEvents()
    assert widget._atom_table.width() == dragged

    # Another host opens the pane at its opening width again.
    widget.set_structure(magnetic_silicon_oxide_host())
    qapp.processEvents()
    assert widget._atom_table.width() == _opening_width(widget)


def test_clearing_the_structure_keeps_the_dragged_split(qapp) -> None:
    widget = HostStructureWidget()
    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()
    widget._viewer.parentWidget().setSizes([500, 500])
    qapp.processEvents()
    dragged = widget._atom_table.width()

    widget.set_structure(None)
    qapp.processEvents()

    assert widget._atom_table.width() == dragged


def _facts(widget: HostStructureWidget) -> dict[str, str]:
    """The facts card's values by caption, every row together."""
    return {caption: value for row in widget._facts_card.facts() for caption, value in row}


def test_propagation_vector_is_one_row_on_top_of_the_atom_table(qapp) -> None:
    """Heading, then `Propagation vector`, three captioned boxes and a help icon, then the table."""
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1100, 760)
    widget.show()
    qapp.processEvents()
    splitter = widget._viewer.parentWidget()
    pane = widget._atom_table.parentWidget()
    table = widget._atom_table

    assert isinstance(splitter, QSplitter)
    assert pane.parentWidget() is splitter
    heading = next(label for label in pane.findChildren(QLabel) if label.text() == "Magnetic order")
    boxes = (widget._propagation_x, widget._propagation_y, widget._propagation_z)
    help_icon = widget._propagation_help
    assert [box.caption() for box in boxes] == ["a*", "b*", "c*"]
    assert all(pane.isAncestorOf(control) for control in (*boxes, help_icon))
    labels = {label.buddy(): label.text() for label in pane.findChildren(QLabel)}
    assert labels[boxes[0]] == "Propagation vector"
    # One row: every control shares a line, below the heading and above the table.
    row_y = {control.mapTo(pane, control.rect().center()).y() for control in (*boxes, help_icon)}
    assert max(row_y) - min(row_y) <= 1
    table_top = table.mapTo(pane, table.rect().topLeft()).y()
    assert heading.y() < boxes[0].mapTo(pane, boxes[0].rect().topLeft()).y() < table_top
    # The row has the table's edges and is no wider than the table.
    assert pane.width() == table.width()
    help_right = help_icon.mapTo(pane, help_icon.rect().topRight()).x()
    assert help_right <= table.width()
    # The long help text is behind the icon, not on the page.
    assert not [label for label in pane.findChildren(QLabel) if "cos(2π k·R)" in label.text()]
    assert "cos(2π k·R)" in help_icon.toolTip()
    assert help_icon.accessibleName() == "About the propagation vector"
    widget.close()


def test_the_table_pane_is_no_wider_than_the_table_needs(qapp) -> None:
    """The propagation row fits within the table's own width at its content size."""
    widget = HostStructureWidget()
    widget.set_structure(cof2_host())

    assert widget._table_pane.minimumSizeHint().width() <= table_natural_width(widget._atom_table)


def test_facts_card_states_contents_space_group_source_and_lattice(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(cof2_host(), source="imports/CoF2.cif")

    assert widget._facts_card.facts() == [
        [
            ("Formula", "Co2F4"),
            ("Atoms", "6"),
            ("Space group", "P4₂/mnm (136)"),
            ("Source", "imports/CoF2.cif"),
        ],
        [
            ("a", "4.6950 Å"),
            ("b", "4.6950 Å"),
            ("c", "3.1800 Å"),
            ("α", "90°"),
            ("β", "90°"),
            ("γ", "90°"),
            ("Volume", "70.10 Å³"),
        ],
    ]
    value = widget._facts_card.value_label("a")
    assert value.accessibleDescription() == "a, 4.6950 Å"
    assert value.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert widget._facts_card.accessibleName() == "Host structure facts"
    assert not widget.findChildren(QGroupBox)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {},
            {"Formula": "SiO", "Atoms": "2", "Space group": "Pm (6)", "Source": "imports/host.cif"},
        ),
        (
            {"source_file": None, "label": "Relaxed host"},
            {"Formula": "SiO", "Atoms": "2", "Space group": "Pm (6)", "Source": "Relaxed host"},
        ),
        (
            {"source_file": None, "species": ("Si",), "positions": ((0.0, 0.0, 0.0),)},
            {"Formula": "Si", "Atoms": "1", "Space group": "Pmmm (47)", "Source": "Host"},
        ),
        (
            # spglib cannot classify two atoms on one site; the group is left out.
            {"species": ("Si", "Si"), "positions": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))},
            {"Formula": "Si2", "Atoms": "2", "Source": "imports/host.cif"},
        ),
    ],
)
def test_facts_card_falls_back_to_the_source_file_or_label(qapp, overrides, expected) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host(**overrides))

    assert dict(widget._facts_card.facts()[0]) == expected


def test_facts_card_without_a_structure(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.set_structure(None)

    assert widget._facts_card.message() == "No host structure loaded."
    assert widget._facts_card.facts() == []


def test_every_fact_shows_whole_in_the_narrowest_window(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(cof2_host(), source="imports/CoF2.cif")
    widget.resize(620, 760)
    widget.show()
    qapp.processEvents()
    card = widget._facts_card

    for caption in ("Source", "Volume"):
        value = card.value_label(caption)
        assert value.mapTo(card, value.rect().topRight()).x() < card.width()
        assert value.width() >= value.sizeHint().width()
    widget.close()


def test_space_group_search_runs_only_when_the_geometry_changes(qapp, monkeypatch) -> None:
    """A spin edit, a refresh or a new source keeps the facts; a new geometry finds them again."""
    from mufinder.app import host_structure_widget as module

    searched: list[object] = []

    def counting_facts(structure):
        searched.append(structure)
        return structure_facts(structure)

    monkeypatch.setattr(module, "structure_facts", counting_facts)
    widget = HostStructureWidget()
    host = magnetic_silicon_oxide_host()

    widget.set_structure(host)
    widget.set_structure(host)
    host.properties["atomic_spins"] = [2.0, -2.0]
    widget.set_structure(host)
    widget.set_structure(host, source="Pristine relaxation relax_1")
    assert len(searched) == 1
    assert _facts(widget)["Source"] == "Pristine relaxation relax_1"

    host.lattice_vectors[0][0] = 5.5
    widget.set_structure(host)
    assert len(searched) == 2
    assert _facts(widget)["a"] == "5.5000 Å"


def test_active_host_selector_is_sized_to_its_entries_with_help_under_it(qapp) -> None:
    widget = HostStructureWidget()
    host = magnetic_silicon_oxide_host()
    widget.set_promotable_hosts(
        [
            PromotableHost(
                structure_id=host.id,
                label="Original — host",
                provenance_hint="Imported from imports/host.cif",
                kind="imported",
                created_at=None,
            )
        ],
        host.id,
    )
    widget.set_structure(host)
    widget.resize(1100, 760)
    widget.show()
    qapp.processEvents()

    combo = widget._active_host_combo
    assert combo.width() < widget.width() // 3
    assert combo.toolTip() == "Original — host"
    help_texts = [label.text() for label in widget.findChildren(QLabel)]
    assert (
        "Every later stage uses this structure. Spins and saved supercells are kept per host."
        in help_texts
    )
    widget.close()


def test_host_structure_widget_gives_spare_height_to_the_table_and_viewer(qapp) -> None:
    """Extra height goes to the table/viewer splitter, not to gaps between the header rows."""
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1100, 1000)
    widget.show()
    qapp.processEvents()

    splitter = widget._viewer.parentWidget()
    assert splitter.height() > widget.height() // 2


def _supports_roll(axis) -> bool:
    return hasattr(axis, "roll")


def test_host_structure_widget_renders_spin_column_and_viewer(qapp) -> None:
    widget = HostStructureWidget()

    widget.set_structure(magnetic_silicon_oxide_host())

    assert widget._atom_table.columnCount() == 5
    headers = [widget._atom_table.horizontalHeaderItem(column) for column in range(5)]
    assert [header.text() for header in headers] == ["Element", "x", "y", "z", "Spin (ħ/2)"]
    assert [header.toolTip() for header in headers[1:4]] == [
        "Fractional coordinate along a.",
        "Fractional coordinate along b.",
        "Fractional coordinate along c.",
    ]
    assert widget._atom_table.item(1, 3).text() == "0.7500"
    assert widget._atom_table.item(1, 4).text() == "-1.5000"
    assert widget._atom_table.verticalHeaderItem(1).text() == "2"
    assert widget._atom_table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert widget._viewer._atoms is not None
    assert "Right drag: rotate" in widget._viewer._interaction_hint.text()
    assert "Middle drag: pan" in widget._viewer._interaction_hint.text()
    assert widget._viewer._selection_label.text() == "Selected atom: -"
    assert widget._viewer._show_spins.isChecked()
    assert not widget._viewer._show_bonds.isChecked()
    assert widget._supercell_button.isEnabled()
    assert widget._propagation_x.value() == 0.0
    assert widget._propagation_z.value() == 0.25

    widget.close()


def test_host_structure_widget_syncs_table_selection_into_viewer(qapp) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())

    widget._atom_table.selectRow(1)

    assert "#2 O" in widget._viewer._selection_label.text()
    assert "0.7500" in widget._viewer._selection_label.text()

    widget._on_viewer_atoms_selected([0])

    assert widget._atom_table.currentRow() == 0

    assert widget._viewer._view_x_button.text() == "a"
    assert widget._viewer._view_y_button.text() == "b"
    assert widget._viewer._view_z_button.text() == "c"

    widget._viewer._set_named_view("b")
    assert widget._viewer._triad_axis is not None
    assert widget._viewer._axis is not None
    assert widget._viewer._triad_axis.elev == widget._viewer._axis.elev
    assert widget._viewer._triad_axis.azim == widget._viewer._axis.azim

    widget._viewer._axis.view_init(elev=31.0, azim=77.0)
    if _supports_roll(widget._viewer._axis):
        widget._viewer._axis.view_init(elev=31.0, azim=77.0, roll=23.0)
    widget._viewer._store_view()
    assert widget._viewer._triad_axis.elev == widget._viewer._axis.elev
    assert widget._viewer._triad_axis.azim == widget._viewer._axis.azim
    if _supports_roll(widget._viewer._axis):
        assert widget._viewer._triad_axis.roll == widget._viewer._axis.roll

    widget.close()


def test_host_structure_widget_syncs_multi_row_table_selection_into_viewer(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    selection_model = widget._atom_table.selectionModel()
    assert selection_model is not None
    selection_model.select(
        widget._atom_table.model().index(0, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    selection_model.select(
        widget._atom_table.model().index(1, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    assert widget._viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in widget._viewer._selection_label.text()
    widget.close()


def test_host_structure_widget_syncs_viewer_multi_selection_into_table(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    widget._on_viewer_atoms_selected([0, 1])

    selection_model = widget._atom_table.selectionModel()
    assert selection_model is not None
    selected_rows = sorted(index.row() for index in selection_model.selectedRows())

    assert selected_rows == [0, 1]
    assert widget._atom_table.currentRow() == 1
    widget.close()


def test_host_structure_widget_blank_click_in_viewer_clears_table_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    widget._atom_table.selectRow(1)
    assert widget._viewer._selected_indexes == [1]

    install_fake_picker(widget._viewer, monkeypatch, picked_point=None, position=(9, 11))
    widget._viewer._on_pyvista_left_button_press()
    widget._viewer._on_pyvista_left_button_release()

    selection_model = widget._atom_table.selectionModel()
    assert selection_model is not None
    assert selection_model.selectedRows() == []
    assert widget._atom_table.currentRow() == -1
    widget.close()


def test_host_structure_widget_emits_magnetic_configuration_on_spin_and_propagation_change(
    qapp,
) -> None:
    widget = HostStructureWidget()
    payloads: list[dict] = []
    widget.magnetic_configuration_changed.connect(payloads.append)
    widget.set_structure(magnetic_silicon_oxide_host())

    spin_item = widget._atom_table.item(1, 4)
    assert spin_item is not None
    spin_item.setText("-2.2500")

    assert payloads[-1]["atomic_spins"] == pytest.approx([1.0, -2.25])

    widget._propagation_x.setValue(0.125)

    latest = payloads[-1]
    # The widget emits spins and k only; ProjectService builds the magnetic structure.
    assert set(latest) == {"atomic_spins", "propagation_vector"}
    assert latest["atomic_spins"] == pytest.approx([1.0, -2.25])
    assert latest["propagation_vector"] == pytest.approx([0.125, 0.0, 0.25])

    widget.close()


def test_host_structure_widget_lists_every_atom_and_edits_the_last_spin(qapp) -> None:
    atom_count = 250
    structure = host_structure(
        lattice=20.0,
        species=["Fe"] * atom_count,
        positions=[(index / atom_count, 0.0, 0.0) for index in range(atom_count)],
        properties={"atomic_spins": [0.0] * atom_count},
    )
    widget = HostStructureWidget()
    payloads: list[dict] = []
    widget.magnetic_configuration_changed.connect(payloads.append)
    widget.set_structure(structure)

    table = widget._atom_table
    assert table.rowCount() == atom_count
    last = atom_count - 1
    assert table.verticalHeaderItem(last).text() == str(atom_count)
    assert table.item(last, 1).text() == f"{last / atom_count:.4f}"
    spin_item = table.item(last, 4)
    assert spin_item.flags() & Qt.ItemFlag.ItemIsEditable

    spin_item.setText("2.5")

    assert payloads[-1]["atomic_spins"][last] == pytest.approx(2.5)
    assert table.item(last, 4).text() == "2.5000"

    emitted = len(payloads)
    spin_item.setText("not a number")

    assert table.item(last, 4).text() == "2.5000"
    assert len(payloads) == emitted

    widget._on_viewer_atoms_selected([last])

    assert [index.row() for index in table.selectionModel().selectedRows()] == [last]

    widget.close()


def test_host_structure_widget_defaults_to_zero_spins_for_invalid_source(qapp) -> None:
    structure = magnetic_silicon_oxide_host()
    structure.properties["atomic_spins"] = "invalid"

    widget = HostStructureWidget()
    widget.set_structure(structure)

    assert widget._viewer._spins is not None
    assert np.allclose(widget._viewer._spins, [0.0, 0.0])

    widget.close()


def test_host_structure_widget_loads_spins_from_magnetic_structure_fallback(qapp) -> None:
    structure = magnetic_silicon_oxide_host()
    structure.properties.pop("atomic_spins", None)
    structure.properties.pop("propagation_vector", None)
    structure.properties["magnetic_structure"] = {
        "propagation_vector": [0.0, 0.5, 0.25],
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 1.25], "coordinate_system": "cartesian"},
            {"label": "O2", "vector": [0.0, 0.0, -0.75], "coordinate_system": "cartesian"},
        ],
    }

    widget = HostStructureWidget()
    widget.set_structure(structure)

    assert widget._viewer._spins is not None
    assert np.allclose(widget._viewer._spins, [1.25, -0.75])
    assert widget._atom_table.item(0, 4).text() == "1.2500"
    assert widget._atom_table.item(1, 4).text() == "-0.7500"
    assert (
        widget._propagation_x.value(),
        widget._propagation_y.value(),
        widget._propagation_z.value(),
    ) == (0.0, 0.5, 0.25)
    widget.close()


def test_host_structure_widget_reads_a_short_propagation_vector_like_the_services(qapp) -> None:
    structure = magnetic_silicon_oxide_host()
    structure.properties["propagation_vector"] = [0.0, 0.5]

    widget = HostStructureWidget()
    widget.set_structure(structure)

    shown = [
        widget._propagation_x.value(),
        widget._propagation_y.value(),
        widget._propagation_z.value(),
    ]
    assert shown == ProjectService().structure_propagation_vector(structure) == [0.0, 0.5, 0.0]
    widget.close()


def test_host_structure_widget_prefers_moments_when_atomic_spins_are_zero(qapp) -> None:
    structure = magnetic_silicon_oxide_host()
    structure.properties["atomic_spins"] = [0.0, 0.0]
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 2.25], "coordinate_system": "cartesian"},
            {"label": "O2", "vector": [0.0, 0.0, -1.25], "coordinate_system": "cartesian"},
        ]
    }

    widget = HostStructureWidget()
    widget.set_structure(structure)

    assert widget._viewer._spins is not None
    assert np.allclose(widget._viewer._spins, [2.25, -1.25])
    widget.close()


def test_host_structure_widget_auto_enables_spin_rendering_on_nonzero_edit(qapp) -> None:
    widget = HostStructureWidget()
    structure = magnetic_silicon_oxide_host()
    structure.properties["atomic_spins"] = [0.0, 0.0]
    widget.set_structure(structure)

    assert not widget._viewer._show_spins.isChecked()

    spin_item = widget._atom_table.item(0, 4)
    assert spin_item is not None
    spin_item.setText("0.5000")
    widget._on_table_cell_changed(0, 4)

    assert widget._viewer._show_spins.isChecked()
    widget.close()


def test_set_structure_skips_viewer_rebuild_when_unchanged(qapp, monkeypatch) -> None:
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    assert len(calls) == 1
    assert widget._atom_table.rowCount() == 2

    # Same structure re-pushed on a flush must not rebuild table or viewer.
    widget.set_structure(host)
    assert len(calls) == 1

    # Different structure id must re-render.
    widget.set_structure(hexagonal_magnesium_oxide_host())
    assert len(calls) == 2


def test_set_structure_none_transitions_render(qapp, monkeypatch) -> None:
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    widget.set_structure(None)
    widget.set_structure(None)  # skipped
    widget.set_structure(host)
    assert calls == [host, None, host]
    assert _facts(widget)["Source"] == "imports/host.cif"


def test_set_structure_rerenders_after_geometry_swap(qapp, monkeypatch) -> None:
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    assert len(calls) == 1

    host.species = ["Si", "O", "H"]
    host.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.5, 0.75], [0.5, 0.5, 0.5]]
    widget.set_structure(host)
    assert len(calls) == 2
    assert widget._atom_table.rowCount() == 3


def test_set_structure_rerenders_after_in_place_spin_edit(qapp, monkeypatch) -> None:
    # atomic_spins live in the in-place-mutated properties dict; the host
    # fingerprint folds them in so a spin change still refreshes.
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    assert len(calls) == 1

    host.properties["atomic_spins"] = [2.0, -2.0]
    widget.set_structure(host)
    assert len(calls) == 2

    # Mutating the *same* spins list in place (identical list identity) must
    # also invalidate the fingerprint — the properties snapshot is content-based.
    spins_id = id(host.properties["atomic_spins"])
    host.properties["atomic_spins"][0] = 3.0
    assert id(host.properties["atomic_spins"]) == spins_id
    widget.set_structure(host)
    assert len(calls) == 3


def test_an_atom_colour_override_redraws_the_viewer_keeping_view_and_table(
    qapp, monkeypatch
) -> None:
    # The viewer draws atom colour overrides, so they are part of what the page
    # compares; the table does not show them, so it and the view are kept.
    widget = HostStructureWidget()
    loads: list[object] = []
    redraws: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: loads.append(structure))
    monkeypatch.setattr(
        widget._viewer,
        "set_structure_preserving_view",
        lambda structure: redraws.append(structure),
    )

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    widget.set_structure(host)
    assert (len(loads), len(redraws)) == (1, 0)
    table_item = widget._atom_table.item(0, 0)

    host.properties["atom_colors"] = ["#ff0000"] + [None] * (len(host.species) - 1)
    widget.set_structure(host)

    assert (len(loads), len(redraws)) == (1, 1)
    assert widget._atom_table.item(0, 0) is table_item
    widget.set_structure(host)
    assert (len(loads), len(redraws)) == (1, 1)
    # A forced load still reloads.
    widget.set_structure(host, force=True)
    assert len(loads) == 2


def test_a_colour_override_keeps_hidden_spins_drawn_from_fallback_moments(qapp) -> None:
    # Spins read from initial_magnetic_moments are not in atomic_spins, which the
    # viewer reads on a reload; the Spins choice the user made still holds.
    host = host_structure(
        species=("Fe", "Fe"), properties={"initial_magnetic_moments": [1.0, -1.0]}
    )
    widget = HostStructureWidget()
    widget.set_structure(host)
    viewer = widget._viewer
    assert viewer._show_spins.isChecked()
    viewer._show_spins.setChecked(False)

    host.properties["atom_colors"] = ["#ff0000", None]
    widget.set_structure(host)

    assert not viewer._show_spins.isChecked()
    assert list(viewer._spins) == [1.0, -1.0]


def test_set_structure_rerenders_after_in_place_element_mutation(qapp, monkeypatch) -> None:
    # Regression for the id()-recycling hazard: the fingerprint is content-based,
    # so mutating geometry *elements* while keeping the very same list objects
    # (identical identities) must still invalidate the guard and re-render.
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = magnetic_silicon_oxide_host()
    widget.set_structure(host)
    assert len(calls) == 1

    species_id = id(host.species)
    positions_id = id(host.fractional_positions)
    lattice_id = id(host.lattice_vectors)
    host.species[1] = "N"
    host.fractional_positions[0][2] = 0.5
    host.lattice_vectors[2][2] = 7.5
    assert id(host.species) == species_id
    assert id(host.fractional_positions) == positions_id
    assert id(host.lattice_vectors) == lattice_id

    widget.set_structure(host)
    assert len(calls) == 2
    assert widget._atom_table.item(1, 0).text() == "N"


def test_host_controls_are_named_or_labelled(qapp) -> None:
    widget = HostStructureWidget()
    labels = {label.buddy(): label.text() for label in widget.findChildren(QLabel)}

    assert labels[widget._active_host_combo] == "Active host"
    assert widget._load_button.text() == "Load Structure…"
    assert widget._atom_table.accessibleName() == "Atoms"
    assert [
        edit.accessibleName()
        for edit in (widget._propagation_x, widget._propagation_y, widget._propagation_z)
    ] == ["Propagation Vector a*", "Propagation Vector b*", "Propagation Vector c*"]


def test_columns_reach_the_right_edge_when_no_scrollbar_shows(main_window, tmp_path: Path) -> None:
    """At the default window size there is no blank strip right of the Spin column."""
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "edge.mufinder", "Edge")
    window._refresh_ui_after_project_change()
    page = window._host_structure_view
    page.set_structure(cof2_host())
    select_route(window, "host_structure")
    window.resize(1280, 864)
    window.show()
    QApplication.processEvents()

    table = page._atom_table
    last = table.columnCount() - 1
    assert not table.verticalScrollBar().isVisible()
    assert table.columnViewportPosition(last) + table.columnWidth(last) == table.viewport().width()
    _assert_atom_table_fits(page)
    window.close()


def test_with_many_rows_the_scrollbar_takes_the_room_and_the_last_column_stays_whole(
    qapp,
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_crowded_host(300))
    widget.resize(1000, 500)
    widget.show()
    qapp.processEvents()

    table = widget._atom_table
    assert table.verticalScrollBar().isVisible()
    _assert_atom_table_fits(widget)
    # The seeded split still measures the content width, not the filled width.
    assert widget._table_pane.width() == _opening_width(widget)
    widget.close()


def test_the_help_icon_opens_a_bubble_from_the_mouse_and_the_keyboard(qapp) -> None:
    widget = HostStructureWidget()
    widget.resize(1100, 760)
    widget.show()
    qapp.processEvents()
    icon = widget._propagation_help

    assert icon.focusPolicy() & Qt.FocusPolicy.TabFocus
    icon.click()
    qapp.processEvents()
    bubble = icon.bubble()
    assert bubble is not None
    assert bubble.text() == icon.toolTip()
    assert "0 0 0" in bubble.text()
    bubble.close()
    qapp.processEvents()
    assert icon.bubble() is None

    icon.setFocus()
    QTest.keyClick(icon, Qt.Key.Key_Return)
    qapp.processEvents()
    assert icon.bubble() is not None
    icon.bubble().close()
    widget.close()


def test_columns_fill_again_after_a_host_with_wider_columns(qapp) -> None:
    widget = HostStructureWidget()
    widget.resize(1100, 760)
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host(species=("Si:labelled_species", "O")))
    qapp.processEvents()
    widget.set_structure(cof2_host())
    qapp.processEvents()

    table = widget._atom_table
    last = table.columnCount() - 1
    assert not table.horizontalScrollBar().isVisible()
    assert table.columnViewportPosition(last) + table.columnWidth(last) == table.viewport().width()
    widget.close()


def test_the_help_bubble_does_not_replay_the_click_that_closes_it(qapp) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget._propagation_help.click()
    qapp.processEvents()
    bubble = widget._propagation_help.bubble()

    assert bubble.testAttribute(Qt.WidgetAttribute.WA_NoMouseReplay)
    assert bubble.windowFlags() & Qt.WindowType.Popup
    bubble.close()
    widget.close()


def test_at_the_default_size_the_propagation_row_is_one_line_with_the_tables_edges(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "row.mufinder", "Row")
    window._refresh_ui_after_project_change()
    page = window._host_structure_view
    page.set_structure(cof2_host())
    select_route(window, "host_structure")
    window.resize(1280, 864)
    window.show()
    QApplication.processEvents()

    pane = page._table_pane
    table = page._atom_table
    controls = (
        page._propagation_label,
        page._propagation_x,
        page._propagation_y,
        page._propagation_z,
        page._propagation_help,
    )
    centres = [c.mapTo(pane, c.rect().center()).y() for c in controls]
    assert max(centres) - min(centres) <= 1
    # Every box shows the longest value it takes.
    for box in controls[1:4]:
        assert box.width() == box.maximumWidth()
    help_right = page._propagation_help.mapTo(pane, page._propagation_help.rect().topRight()).x()
    table_right = table.mapTo(pane, table.rect().topRight()).x()
    assert help_right == table_right
    assert table.mapTo(pane, table.rect().topLeft()).x() == page._propagation_label.x()
    window.close()


def test_the_help_icon_is_quiet_and_no_taller_than_the_boxes(qapp) -> None:
    widget = HostStructureWidget()
    widget.resize(1100, 760)
    widget.show()
    qapp.processEvents()
    icon = widget._propagation_help
    box = widget._propagation_x

    assert icon.objectName() == "mufinderInlineHelpButton"
    assert icon.iconSize().height() <= icon.fontMetrics().height() + 2
    assert icon.height() <= box.height()
    icon_centre = icon.mapTo(widget, icon.rect().center()).y()
    box_centre = box.mapTo(widget, box.rect().center()).y()
    assert abs(icon_centre - box_centre) <= 1
    widget.close()


def test_the_facts_card_spans_the_page_and_has_no_frame_with_no_host(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(cof2_host(), source="imports/CoF2.cif")
    widget.resize(1400, 760)
    widget.show()
    qapp.processEvents()
    card = widget._facts_card
    splitter = widget._splitter

    assert card.x() == splitter.x()
    assert card.width() == splitter.width()
    assert card.property("factCardState") == "facts"

    widget.set_structure(None)
    qapp.processEvents()
    assert card.property("factCardState") == "message"
    assert card.layout().contentsMargins().left() == 0
    widget.close()


def test_at_the_default_size_every_fact_shares_one_line_with_a_hairline_before_a(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "facts.mufinder", "Facts")
    window._refresh_ui_after_project_change()
    page = window._host_structure_view
    page.set_structure(cof2_host(), source="imports/CoF2.cif")
    select_route(window, "host_structure")
    window.resize(1280, 864)
    window.show()
    QApplication.processEvents()
    card = page._facts_card

    def left(caption: str) -> int:
        label = card.value_label(caption)
        return label.mapTo(card, label.rect().topLeft()).x()

    def top(caption: str) -> int:
        label = card.value_label(caption)
        return label.mapTo(card, label.rect().topLeft()).y()

    captions = [caption for group in card.facts() for caption, _value in group]
    assert captions == [
        "Formula",
        "Atoms",
        "Space group",
        "Source",
        "a",
        "b",
        "c",
        "α",
        "β",
        "γ",
        "Volume",
    ]
    assert card.is_one_line()
    assert len({top(caption) for caption in captions}) == 1
    (hairline,) = card.hairlines()
    hairline_x = hairline.mapTo(card, hairline.rect().topLeft()).x()
    assert hairline.isVisible()
    assert left("Source") < hairline_x < left("a")
    # The card has the splitter's edges.
    assert card.mapTo(page, card.rect().topLeft()).x() == page._splitter.x()
    assert card.width() == page._splitter.width()
    window.close()


def test_at_the_pages_minimum_width_the_facts_take_two_rows_and_no_hairline(
    designed_text_scale,
) -> None:
    qapp = designed_text_scale
    widget = HostStructureWidget()
    widget.set_structure(cof2_host(), source="imports/CoF2.cif")
    widget.resize(620, 760)
    widget.show()
    qapp.processEvents()
    card = widget._facts_card

    def top(caption: str) -> int:
        label = card.value_label(caption)
        return label.mapTo(card, label.rect().topLeft()).y()

    assert widget.width() == 620
    assert not card.is_one_line()
    assert not card.hairlines()[0].isVisible()
    assert top("a") > top("Source")
    widget.close()


def test_a_host_without_a_space_group_has_one_fact_fewer(qapp) -> None:
    widget = HostStructureWidget()
    overlapping = magnetic_silicon_oxide_host(
        species=("Si", "Si"), positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    widget.set_structure(overlapping)

    captions = [caption for group in widget._facts_card.facts() for caption, _ in group]
    assert "Space group" not in captions
    assert len(captions) == 10


def test_a_very_long_source_is_shortened_and_the_facts_stay_on_one_line(qapp) -> None:
    path = "imports/" + "a_very_long_structure_file_name_" * 5 + "end.cif"
    widget = HostStructureWidget()
    widget.set_structure(cof2_host(), source=path)
    widget.resize(1400, 760)
    widget.show()
    qapp.processEvents()
    card = widget._facts_card
    value = card.value_label("Source")

    assert card.is_one_line()
    assert "…" in value.text()
    assert value.toolTip() == path
    assert widget.width() == 1400
    widget.close()


def test_columns_are_measured_again_after_a_larger_text_size(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1100, 760)
    widget.show()
    qapp.processEvents()

    widget.setStyleSheet("QTableView, QHeaderView { font-size: 30px; }")
    qapp.processEvents()

    _assert_atom_table_fits(widget)
    widget.close()


def test_a_new_source_name_updates_the_facts_without_rebuilding_the_viewer(
    qapp, monkeypatch
) -> None:
    widget = HostStructureWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))
    host = cof2_host()
    widget.set_structure(host, source="Pristine relaxation relax_1")

    widget.set_structure(host, source="Pristine relaxation pbe_relax_relax_1")

    assert len(calls) == 1
    assert _facts(widget)["Source"] == "Pristine relaxation pbe_relax_relax_1"


def _shown_host_page_with_render_windows(qapp, monkeypatch) -> HostStructureWidget:
    """Host Structure with a structure, shown and active, its viewer on production canvases.

    The canvases are the viewer's own, each holding a ``FakePlotter`` with the focus
    policy of the real ``QtInteractor``, so the Tab path is the one the app has.
    """
    interactors = count_created_interactors(monkeypatch)
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    widget.resize(1100, 760)
    widget.show()
    widget.activateWindow()
    assert QTest.qWaitForWindowActive(widget)
    qapp.processEvents()
    viewer = widget._viewer
    assert {viewer._canvas.plotter, viewer._triad_canvas.plotter} == set(interactors)
    return widget


def _in_render_area(viewer, focus) -> bool:
    """Whether ``focus`` is the viewer's scene or triad canvas, or inside either."""
    return any(
        focus is canvas or (focus is not None and canvas.isAncestorOf(focus))
        for canvas in (viewer._canvas, viewer._triad_canvas)
    )


@pytest.mark.parametrize(
    "modifiers", [Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier]
)
def test_tab_goes_round_host_structure_without_stopping_in_the_viewer(
    qapp, monkeypatch, modifiers
) -> None:
    widget = _shown_host_page_with_render_windows(qapp, monkeypatch)
    viewer = widget._viewer
    try:
        start = viewer._show_cell
        start.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()
        assert QApplication.focusWidget() is start

        stops = []
        for _ in range(200):
            stops.append(press_key(Qt.Key.Key_Tab, modifiers))
            if stops[-1] is start:
                break
        assert stops[-1] is start, "Tab did not come back round to the Cell box"

        assert not [stop for stop in stops if _in_render_area(viewer, stop)]
        # The viewer's own controls are still stops, in their on-screen order.
        toolbar = [
            viewer._show_cell,
            viewer._show_bonds,
            viewer._show_spins,
            viewer._view_x_button,
            viewer._view_y_button,
            viewer._view_z_button,
            viewer._reset_view_button,
        ]
        round_trip = [start, *stops[:-1]]
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            round_trip = [start, *reversed(round_trip[1:])]
        assert [stop for stop in round_trip if stop in toolbar] == toolbar
    finally:
        widget.close()


def test_a_click_in_the_viewer_gives_it_focus_and_it_still_sees_shift(qapp, monkeypatch) -> None:
    widget = _shown_host_page_with_render_windows(qapp, monkeypatch)
    viewer = widget._viewer
    plotter = viewer._canvas.plotter
    try:
        widget._atom_table.setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()

        QTest.mouseClick(plotter, Qt.MouseButton.LeftButton, pos=plotter.rect().center())
        qapp.processEvents()
        assert QApplication.focusWidget() is plotter

        # A key goes to whatever has focus; the viewer reads Shift from the keys it
        # gets, for Shift-click selection. One event each way, as a real Shift key
        # sends (``QTest`` would add a modifier press of its own). Each carries the
        # modifiers held before it, as the platform reports them; Qt flips Shift in
        # ``modifiers()`` for the Shift key itself, so the press reads as Shift held.
        for event_type, modifiers_before, held in (
            (QEvent.Type.KeyPress, Qt.KeyboardModifier.NoModifier, True),
            (QEvent.Type.KeyRelease, Qt.KeyboardModifier.ShiftModifier, False),
        ):
            key = QKeyEvent(event_type, Qt.Key.Key_Shift, modifiers_before)
            QApplication.sendEvent(QApplication.focusWidget(), key)
            assert viewer._native_selection_modifier is held
    finally:
        release_keyboard_modifiers()
        widget.close()
