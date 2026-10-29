"""Tests for the Host Structure page: layout, atom table, spins and selection sync.

The structure viewer's own behaviour is tested in ``test_atomic_structure_viewer.py``
(selection, camera, theme, shutdown) and ``test_atomic_structure_viewer_scene.py``
(meshes and overlays).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt

from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.services.project_service import ProjectService
from tests.support.pyvista_fakes import install_fake_picker
from tests.support.structures import (
    hexagonal_magnesium_oxide_host,
    host_structure,
    magnetic_silicon_oxide_host,
)


def test_host_structure_widget_splitter_fits_small_screen_width(qapp) -> None:
    """Viewer/table minimums fit a 1280px window (SC-7 small-screen fix).

    The table is capped at its content width (never locked to it), so the
    central splitter can shrink the table pane below its content width with a
    horizontal scrollbar, leaving room for the reduced-minimum viewer.
    """
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    assert widget._viewer.minimumWidth() == 380
    assert widget._atom_table.minimumWidth() < widget._atom_table.maximumWidth()

    splitter = widget._viewer.parentWidget()
    from PySide6.QtWidgets import QSplitter

    assert isinstance(splitter, QSplitter)
    # Combined minimum width (viewer + table pane) must comfortably fit a
    # 1280px-wide window; the previous 560 viewer minimum alone left too
    # little room once the table's content-width lock was added on top.
    assert widget._viewer.minimumWidth() + widget._atom_table.minimumWidth() < 1280


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
    assert widget._atom_table.horizontalHeaderItem(3).text() == "fZ"
    assert widget._atom_table.horizontalHeaderItem(4).text() == "Spin (hbar/2)"
    assert widget._atom_table.item(1, 3).text() == "0.7500"
    assert widget._atom_table.item(1, 4).text() == "-1.5000"
    assert widget._atom_table.verticalHeaderItem(1).text() == "2"
    assert widget._atom_table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    # The table is capped at its content width (so it cannot greedily expand)
    # but the minimum is deliberately left low, so the splitter can shrink it
    # below content width with a horizontal scrollbar on small screens.
    assert widget._atom_table.maximumWidth() > 0
    assert widget._atom_table.minimumWidth() < widget._atom_table.maximumWidth()
    assert widget._atom_table.maximumWidth() >= widget._propagation_controls_width()
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
    widget._on_table_cell_changed(1, 4)

    widget._propagation_x.setValue(0.125)
    widget._on_propagation_changed()

    assert payloads
    latest = payloads[-1]
    assert latest["atomic_spins"][1] == pytest.approx(-2.25)
    assert latest["propagation_vector"][0] == pytest.approx(0.125)
    assert latest["magnetic_structure"] is not None
    assert latest["magnetic_structure"]["propagation_vector"][2] == pytest.approx(0.25)

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
    assert widget._summary.text().startswith("Loaded structure")


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
