"""Tests for the host structure widget and embedded structure viewer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
import numpy as np

from mufinder.app.atomic_structure_viewer import _draw_spin_arrows, _draw_split_bond_cylinder, _set_equal_limits
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.domain.models import Structure


def _sample_structure() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        species=["Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]],
        source_file="imports/host.cif",
        properties={
            "pbc": [True, True, True],
            "atomic_spins": [1.0, -1.5],
            "propagation_vector": [0.0, 0.0, 0.25],
        },
    )


def _supports_roll(axis) -> bool:
    return hasattr(axis, "roll")


def test_host_structure_widget_renders_spin_column_and_viewer(qapp) -> None:
    widget = HostStructureWidget()

    widget.set_structure(_sample_structure())

    assert widget._atom_table.columnCount() == 5
    assert widget._atom_table.horizontalHeaderItem(3).text() == "fZ"
    assert widget._atom_table.horizontalHeaderItem(4).text() == "Spin (hbar/2)"
    assert widget._atom_table.item(1, 3).text() == "0.7500"
    assert widget._atom_table.item(1, 4).text() == "-1.5000"
    assert widget._atom_table.verticalHeaderItem(1).text() == "2"
    assert (
        widget._atom_table.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert widget._atom_table.minimumWidth() > 0
    assert widget._atom_table.minimumWidth() == widget._atom_table.maximumWidth()
    assert widget._atom_table.minimumWidth() >= widget._propagation_controls_width()
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
    widget.set_structure(_sample_structure())

    widget._on_table_selection_changed(1, 0, -1, -1)

    assert "#2 O" in widget._viewer._selection_label.text()
    assert "0.7500" in widget._viewer._selection_label.text()

    widget._on_viewer_atom_selected(0)

    assert widget._atom_table.currentRow() == 0

    widget._viewer._set_named_view("y")
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


def test_host_structure_widget_emits_magnetic_configuration_on_spin_and_propagation_change(qapp) -> None:
    widget = HostStructureWidget()
    payloads: list[dict] = []
    widget.magnetic_configuration_changed.connect(payloads.append)
    widget.set_structure(_sample_structure())

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


def test_atomic_structure_viewer_spin_arrow_geometry_is_centered() -> None:
    figure = Figure()
    axis = figure.add_subplot(projection="3d")

    points = _draw_spin_arrows(
        axis,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        radii=np.array([0.6, 0.45], dtype=float),
        spins=np.array([1.2, -0.8], dtype=float),
    )

    assert points.shape == (4, 3)
    assert np.any(points[:, 2] > 0.0)
    assert np.any(points[:, 2] < 1.0)


def test_split_bond_cylinder_draws_two_half_segments_with_per_atom_colors() -> None:
    class DummyAxis:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plot_surface(self, x, y, z, **kwargs) -> object:
            self.calls.append({"x": x, "y": y, "z": z, "kwargs": kwargs})
            return object()

    axis = DummyAxis()
    start_color = np.array([0.1, 0.2, 0.3], dtype=float)
    end_color = np.array([0.8, 0.4, 0.1], dtype=float)

    segments = _draw_split_bond_cylinder(
        axis,
        start=np.array([0.0, 0.0, 0.0], dtype=float),
        end=np.array([1.6, 0.0, 0.0], dtype=float),
        start_color=start_color,
        end_color=end_color,
        radius=0.12,
        alpha=0.47,
        resolution=11,
    )

    assert segments == 2
    assert len(axis.calls) == 2
    assert np.allclose(axis.calls[0]["kwargs"]["color"], start_color)
    assert np.allclose(axis.calls[1]["kwargs"]["color"], end_color)
    assert axis.calls[0]["kwargs"]["alpha"] == pytest.approx(0.47)
    assert axis.calls[1]["kwargs"]["alpha"] == pytest.approx(0.47)


def test_set_equal_limits_include_atom_radii_to_avoid_clipping() -> None:
    figure = Figure()
    axis = figure.add_subplot(projection="3d")

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    radii = np.array([0.5, 0.9], dtype=float)

    _set_equal_limits(axis, positions, atom_radii=radii, margin=0.0)

    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    assert x_min <= -0.5
    assert x_max >= 2.9
    assert y_min < 0.0 < y_max
    assert x_span > y_span


def test_set_equal_limits_preserves_isotropic_box_aspect_in_wide_viewport() -> None:
    figure = Figure(figsize=(11.0, 5.0))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0], projection="mufinder3d")

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    _set_equal_limits(axis, positions, margin=0.0)

    box_aspect = axis.get_box_aspect()
    assert float(box_aspect[0]) == pytest.approx(float(box_aspect[1]))
    assert float(box_aspect[1]) == pytest.approx(float(box_aspect[2]))


def test_atomic_structure_viewer_controls_default_states_and_reset_button_width(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    assert not viewer._show_bonds.isChecked()
    assert viewer._show_spins.isChecked()
    assert viewer._triad_canvas.width() == 132
    assert viewer._reset_view_button.minimumWidth() >= viewer._reset_view_button.sizeHint().width()

    viewer._show_bonds.setChecked(True)
    assert viewer._show_bonds.isChecked()

    viewer._show_spins.setChecked(False)
    viewer._show_spins.setChecked(True)

    widget.close()


def test_bond_alpha_tracks_atom_transparency_state(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    alphas: list[float] = []

    def _capture(_axis, _positions, _colors, _radii, *, alpha: float) -> None:
        alphas.append(alpha)

    monkeypatch.setattr(viewer, "_draw_bonds", _capture)

    viewer._show_bonds.setChecked(True)
    viewer._show_spins.setChecked(False)
    viewer._show_spins.setChecked(True)

    assert len(alphas) >= 3
    assert alphas[0] < alphas[1]
    assert alphas[-1] == pytest.approx(alphas[0])

    widget.close()


def test_atomic_structure_viewer_handles_empty_structure_and_mismatched_spin_payload(qapp) -> None:
    widget = HostStructureWidget()
    viewer = widget._viewer

    viewer.set_structure(None)
    assert viewer._atoms is None
    assert viewer._triad_axis is None

    viewer.set_spins([1.0, -1.0])
    assert viewer._spins is None

    widget.close()


def test_atomic_structure_viewer_defaults_to_zero_spins_for_invalid_source(qapp) -> None:
    structure = _sample_structure()
    structure.properties["atomic_spins"] = "invalid"

    widget = HostStructureWidget()
    widget.set_structure(structure)

    assert widget._viewer._spins is not None
    assert np.allclose(widget._viewer._spins, [0.0, 0.0])

    widget.close()


def test_host_structure_widget_loads_spins_from_magnetic_structure_fallback(qapp) -> None:
    structure = _sample_structure()
    structure.properties.pop("atomic_spins", None)
    structure.properties["magnetic_structure"] = {
        "propagation_vector": [0.0, 0.0, 0.25],
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
    widget.close()


def test_host_structure_widget_prefers_moments_when_atomic_spins_are_zero(qapp) -> None:
    structure = _sample_structure()
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


def test_atomic_structure_viewer_blank_click_clears_selection(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    viewer.select_atom(1)
    assert "#2 O" in viewer._selection_label.text()

    viewer._on_mouse_press(SimpleNamespace(inaxes=viewer._axis, button=1))

    assert viewer._selection_label.text() == "Selected atom: -"
    widget.close()


def test_atomic_structure_viewer_mouse_move_redraws_triad_canvas(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    calls = {"main": 0, "triad": 0}

    if _supports_roll(viewer._axis):
        viewer._axis.view_init(elev=17.0, azim=52.0, roll=-28.0)

    monkeypatch.setattr(viewer._canvas, "draw_idle", lambda: calls.__setitem__("main", calls["main"] + 1))
    monkeypatch.setattr(
        viewer._triad_canvas,
        "draw_idle",
        lambda: calls.__setitem__("triad", calls["triad"] + 1),
    )

    viewer._on_mouse_move(SimpleNamespace(inaxes=viewer._axis))

    assert calls["main"] == 1
    assert calls["triad"] == 1
    if _supports_roll(viewer._axis):
        assert viewer._triad_axis.roll == pytest.approx(viewer._axis.roll)
    widget.close()


def test_host_structure_widget_auto_enables_spin_rendering_on_nonzero_edit(qapp) -> None:
    widget = HostStructureWidget()
    structure = _sample_structure()
    structure.properties["atomic_spins"] = [0.0, 0.0]
    widget.set_structure(structure)

    assert not widget._viewer._show_spins.isChecked()

    spin_item = widget._atom_table.item(0, 4)
    assert spin_item is not None
    spin_item.setText("0.5000")
    widget._on_table_cell_changed(0, 4)

    assert widget._viewer._show_spins.isChecked()
    widget.close()
