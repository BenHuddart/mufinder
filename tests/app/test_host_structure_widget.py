"""Tests for the host structure widget and embedded structure viewer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QItemSelectionModel, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from matplotlib.figure import Figure
import numpy as np

from mufinder.app.atomic_structure_viewer import (
    AtomicStructureViewer,
    _blend_hex_colors,
    _build_pyvista_atom_glyph_mesh,
    _build_pyvista_bond_tube_mesh,
    _build_pyvista_bond_line_mesh,
    _CameraState,
    _configure_transparent_pyvista_background,
    _draw_atom_spheres,
    _plot_cylinder_segment,
    _shutdown_pyvista_widget,
    _WrappingHintLabel,
    _build_pyvista_sphere_mesh,
    _camera_state_from_vectors,
    _camera_vectors_from_state,
    _draw_spin_arrows,
    _draw_split_bond_cylinder,
    _set_equal_limits,
    _ViewerIsosurfaceSpec,
    _ViewerPointOverlaySpec,
    _ViewerSliceSpec,
    _PyVistaCanvas,
    _viewer_background_color,
    _viewer_style,
    _viewer_style_signature,
)
from mufinder.app import theme as theme_module
from mufinder.app.theme import body_font_family, current_palette
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


def _hexagonal_structure() -> Structure:
    a = 4.2
    c = 6.8
    return Structure(
        label="Hex",
        lattice_vectors=[
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        species=["Mg", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.25]],
        source_file="imports/hex.cif",
        properties={"pbc": [True, True, True]},
    )


def _set_slice_planes_hiding_second_atom(viewer) -> None:
    viewer.set_slice_planes(
        [
            {
                "x_coordinates": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
                "y_coordinates": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
                "z_coordinates": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
                "values": np.zeros((2, 2), dtype=float),
                "anchor": np.array([0.0, 0.0, 0.0], dtype=float),
                "normal": np.array([1.0, 0.0, 0.0], dtype=float),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            {
                "x_coordinates": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
                "y_coordinates": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
                "z_coordinates": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
                "values": np.zeros((2, 2), dtype=float),
                "anchor": np.array([0.0, 0.0, 0.0], dtype=float),
                "normal": np.array([0.0, 1.0, 0.0], dtype=float),
                "minimum": 0.0,
                "maximum": 1.0,
            },
        ]
    )
    viewer.set_hide_atoms_in_front_of_planes(True)


def _supports_roll(axis) -> bool:
    return hasattr(axis, "roll")


def _install_fake_pyvista_picker(
    viewer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    picked_point: tuple[float, float, float] | None,
    shift: bool = False,
    position: tuple[int, int] = (18, 24),
):
    class _FakePicker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, object]] = []

        def Pick(self, x: int, y: int, z: int, renderer) -> int:
            self.calls.append((x, y, z, renderer))
            return 0 if picked_point is None else 1

        def GetPickPosition(self):
            if picked_point is None:
                return (0.0, 0.0, 0.0)
            return picked_point

    picker = _FakePicker()
    current_position = {"value": position}
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(
                interactor=SimpleNamespace(GetShiftKey=lambda: int(shift)),
                picker=picker,
                get_event_position=lambda: current_position["value"],
                get_poked_renderer=lambda: "renderer",
            )
        )
    )
    return picker, current_position


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
    widget.set_structure(_sample_structure())

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
    widget.set_structure(_sample_structure())

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
    widget.set_structure(_sample_structure())

    widget._atom_table.selectRow(1)
    assert widget._viewer._selected_indexes == [1]

    _install_fake_pyvista_picker(widget._viewer, monkeypatch, picked_point=None, position=(9, 11))
    widget._viewer._on_pyvista_left_button_press()
    widget._viewer._on_pyvista_left_button_release()

    selection_model = widget._atom_table.selectionModel()
    assert selection_model is not None
    assert selection_model.selectedRows() == []
    assert widget._atom_table.currentRow() == -1
    widget.close()


def test_atomic_structure_viewer_named_views_follow_lattice_vectors_for_hexagonal_cell(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_hexagonal_structure())

    viewer = widget._viewer
    lattice = np.asarray(viewer._atoms.cell, dtype=float)

    viewer._set_named_view("b")
    position_b, _focal_b, _up_b = _camera_vectors_from_state(viewer._camera_state)
    direction_b = np.asarray(position_b, dtype=float) / np.linalg.norm(position_b)
    assert direction_b == pytest.approx(lattice[1] / np.linalg.norm(lattice[1]))
    assert not np.allclose(direction_b, np.array([0.0, 1.0, 0.0], dtype=float))

    viewer._set_named_view("c")
    position_c, _focal_c, _up_c = _camera_vectors_from_state(viewer._camera_state)
    direction_c = np.asarray(position_c, dtype=float) / np.linalg.norm(position_c)
    assert direction_c == pytest.approx(lattice[2] / np.linalg.norm(lattice[2]))

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
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")

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
    assert viewer.backend_name == "pyvista"

    viewer._show_bonds.setChecked(True)
    assert viewer._show_bonds.isChecked()

    viewer._show_spins.setChecked(False)
    viewer._show_spins.setChecked(True)

    widget.close()


def test_pyvista_canvas_disables_pyvistaqt_auto_update_render_timer(qapp, monkeypatch) -> None:
    """The canvas must build its interactor with ``auto_update=False`` (CPU regression).

    pyvistaqt's ``QtInteractor`` defaults to ``auto_update=5.0``, which starts a
    timer that calls ``render()`` ~5x/sec forever regardless of visibility and
    pegged the CPU even on tabs without the view. We render explicitly, so the
    periodic timer must be disabled at construction.
    """

    import mufinder.app.atomic_structure_viewer as module

    captured: dict[str, object] = {}

    class _RecordingInteractor(module.QWidget):
        def __init__(self, parent=None, **kwargs) -> None:
            super().__init__(parent)
            captured.update(kwargs)

        def set_background(self, *_args, **_kwargs) -> None:
            return

        def enable_parallel_projection(self) -> None:
            return

        def enable_anti_aliasing(self, *_args, **_kwargs) -> None:
            return

    monkeypatch.setattr(module, "_pyvista_modules", lambda: (object(), _RecordingInteractor))

    # _PyVistaCanvas was imported before the autouse fake patched the module
    # attribute, so this constructs the real production canvas.
    canvas = _PyVistaCanvas()
    try:
        assert captured.get("auto_update") is False
    finally:
        canvas.deleteLater()


def test_atomic_structure_viewer_pauses_camera_poll_while_hidden(qapp) -> None:
    """The camera poll timer must stop while hidden and resume when shown again.

    It only needs to tick while the view is visible (to sync axes/triad to
    user-driven camera moves); leaving it running on tabs without the view is
    wasted work.
    """

    from PySide6.QtGui import QHideEvent, QShowEvent

    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    assert viewer._pyvista_camera_timer.isActive()
    viewer.hideEvent(QHideEvent())
    assert not viewer._pyvista_camera_timer.isActive()
    viewer.showEvent(QShowEvent())
    assert viewer._pyvista_camera_timer.isActive()

    widget.close()


def test_atomic_structure_viewer_requires_pyvista_modules(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (None, None))

    with pytest.raises(RuntimeError, match="requires PyVista"):
        AtomicStructureViewer()


def test_atomic_structure_viewer_selection_label_uses_fixed_single_line_height(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    assert viewer._selection_label.wordWrap() is False
    assert viewer._selection_label.minimumHeight() == viewer._selection_label.maximumHeight()

    viewer.select_atom(1)

    assert "#2 O" in viewer._selection_label.text()
    assert viewer._selection_label.toolTip() == viewer._selection_label.text()
    assert viewer._selection_label.minimumHeight() == viewer._selection_label.maximumHeight()
    widget.close()


def test_bond_alpha_tracks_atom_transparency_state(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    alphas: list[float] = []

    def _capture(_plotter, _pv, _positions, _colors, _radii, *, visible_mask=None, alpha: float):
        del visible_mask
        alphas.append(alpha)
        return []

    monkeypatch.setattr(viewer, "_draw_pyvista_bonds", _capture)

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


def test_atomic_structure_viewer_blank_click_clears_selection(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    viewer.select_atom(1)
    assert "#2 O" in viewer._selection_label.text()

    _install_fake_pyvista_picker(viewer, monkeypatch, picked_point=None, position=(9, 11))
    viewer._on_pyvista_left_button_press()
    assert "#2 O" in viewer._selection_label.text()
    viewer._on_pyvista_left_button_release()

    assert viewer._selection_label.text() == "Selected atom: -"
    widget.close()


def test_atomic_structure_viewer_shift_selection_adds_and_removes_atoms(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_selection_modifier_active", lambda _event=None: True)

    viewer._apply_pick_selection(1)
    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()

    viewer._apply_pick_selection(0)
    assert viewer._selected_indexes == [1]
    assert "#2 O" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_shift_selection_uses_latched_modifier_when_pick_event_loses_it(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_selection_modifier_active", lambda _event=None: False)
    viewer._latched_selection_modifier = True

    viewer._apply_pick_selection(1)

    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_shift_selection_uses_native_mouse_modifier_when_backend_event_loses_it(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    viewer.select_atom(0)
    native_target = viewer._canvas
    native_press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    viewer.eventFilter(native_target, native_press)

    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_lookup_indices = np.array([0, 1], dtype=int)
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(interactor=SimpleNamespace(GetShiftKey=lambda: 0))
        )
    )

    assert viewer._selected_indexes == [0]

    viewer._on_pyvista_point_picked(np.array([1.1, 1.0, 1.0], dtype=float))

    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_blank_click_clears_multi_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    viewer.select_atoms([0, 1])
    assert viewer._selected_indexes == [0, 1]

    _install_fake_pyvista_picker(viewer, monkeypatch, picked_point=None, position=(9, 11))
    viewer._on_pyvista_left_button_press()
    assert viewer._selected_indexes == [0, 1]
    viewer._on_pyvista_left_button_release()

    assert viewer._selected_indexes == []
    assert viewer._selection_label.text() == "Selected atom: -"
    widget.close()


def test_camera_state_from_vectors_round_trips_default_view() -> None:
    state = _camera_state_from_vectors(
        position=(5.0, 5.0, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
    )

    assert isinstance(state, _CameraState)
    assert -180.0 <= state.azim <= 180.0
    assert -90.0 <= state.elev <= 90.0


def test_atomic_structure_viewer_pyvista_pick_selects_nearest_atom(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(interactor=SimpleNamespace(GetShiftKey=lambda: 0))
        )
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._on_pyvista_point_picked(np.array([1.1, 1.0, 1.0], dtype=float))

    assert "#2 O" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_pyvista_pick_uses_interactor_shift_state_for_multi_select(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._native_selection_modifier = False
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(interactor=SimpleNamespace(GetShiftKey=lambda: 1))
        )
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_lookup_indices = np.array([0, 1], dtype=int)

    viewer._on_pyvista_point_picked(np.array([1.1, 1.0, 1.0], dtype=float))

    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_pyvista_left_click_release_uses_interactor_shift_state_for_multi_select(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    class _FakePicker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, object]] = []

        def Pick(self, x: int, y: int, z: int, renderer) -> int:
            self.calls.append((x, y, z, renderer))
            return 1

        def GetPickPosition(self):
            return (1.1, 1.0, 1.0)

    current_position = {"value": (18, 24)}
    picker = _FakePicker()
    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._native_selection_modifier = False
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(
                interactor=SimpleNamespace(GetShiftKey=lambda: 1),
                picker=picker,
                get_event_position=lambda: current_position["value"],
                get_poked_renderer=lambda: "renderer",
            )
        )
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_lookup_indices = np.array([0, 1], dtype=int)

    viewer._on_pyvista_left_button_press()
    viewer._on_pyvista_left_button_release()

    assert picker.calls == [(18, 24, 0, "renderer")]
    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_pyvista_left_click_release_clears_selection_on_empty_pick(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    class _FakePicker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, object]] = []

        def Pick(self, x: int, y: int, z: int, renderer) -> int:
            self.calls.append((x, y, z, renderer))
            return 0

    current_position = {"value": (9, 11)}
    picker = _FakePicker()
    viewer.select_atom(1)
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._native_selection_modifier = False
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(
                interactor=SimpleNamespace(GetShiftKey=lambda: 0),
                picker=picker,
                get_event_position=lambda: current_position["value"],
                get_poked_renderer=lambda: "renderer",
            )
        )
    )

    viewer._on_pyvista_left_button_press()
    viewer._on_pyvista_left_button_release()

    assert picker.calls == [(9, 11, 0, "renderer")]
    assert viewer._selected_indexes == []
    assert viewer._selection_label.text() == "Selected atom: -"
    widget.close()


def test_atomic_structure_viewer_pyvista_drag_does_not_trigger_click_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    class _FakePicker:
        def __init__(self) -> None:
            self.calls = 0

        def Pick(self, x: int, y: int, z: int, renderer) -> int:
            del x, y, z, renderer
            self.calls += 1
            return 1

        def GetPickPosition(self):
            return (1.1, 1.0, 1.0)

    current_position = {"value": (5, 7)}
    picker = _FakePicker()
    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(
                interactor=SimpleNamespace(GetShiftKey=lambda: 0),
                picker=picker,
                get_event_position=lambda: current_position["value"],
                get_poked_renderer=lambda: "renderer",
            )
        )
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_lookup_indices = np.array([0, 1], dtype=int)

    viewer._on_pyvista_left_button_press()
    current_position["value"] = (22, 29)
    viewer._on_pyvista_left_button_release()

    assert picker.calls == 0
    assert viewer._selected_indexes == [0]
    assert "#1 Si" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_renders_isosurface_overlay(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    viewer.set_isosurface(
        vertices=[
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
        ],
        faces=[[0, 1, 2]],
    )

    assert viewer._isosurface_vertices is not None
    assert viewer._isosurface_faces is not None
    assert viewer._isosurface_faces.shape == (1, 3)
    viewer.set_isosurface(None, None)
    assert viewer._isosurface_vertices is None
    assert viewer._isosurface_faces is None
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_isosurface_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []
    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"actor-{len(add_calls)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_isosurface_actor = "actor-existing"

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(
        viewer,
        "_build_pyvista_isosurface_mesh",
        lambda _pv, *, vertices, faces: "mesh",
    )
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(viewer, "_render", lambda: pytest.fail("full render should not run for PyVista isosurface-only updates"))

    viewer.set_isosurface(
        vertices=[
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
        ],
        faces=[[0, 1, 2]],
    )

    assert remove_calls == [("actor-existing", False)]
    assert len(add_calls) == 1
    assert add_calls[0]["mesh"] == "mesh"
    assert add_calls[0]["kwargs"]["render"] is False
    assert viewer._pyvista_isosurface_actor == "actor-1"
    assert draw_calls["count"] == 1

    viewer.set_isosurface(None, None)

    assert remove_calls[-1] == ("actor-1", False)
    assert viewer._pyvista_isosurface_actor is None
    assert draw_calls["count"] == 2
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_slice_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []
    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"slice-{len(add_calls)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    class _FakePv:
        @staticmethod
        def StructuredGrid(x_coordinates, y_coordinates, z_coordinates):
            return {
                "x_coordinates": x_coordinates,
                "y_coordinates": y_coordinates,
                "z_coordinates": z_coordinates,
            }

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_slice_actors = ["slice-existing"]

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (_FakePv, None))
    monkeypatch.setattr(viewer, "_render", lambda: pytest.fail("full render should not run for PyVista slice-only updates"))

    viewer.set_slice_planes(
        [
            {
                "x_coordinates": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
                "y_coordinates": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
                "z_coordinates": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
                "values": np.zeros((2, 2), dtype=float),
                "anchor": np.array([0.0, 0.0, 0.0], dtype=float),
                "normal": np.array([1.0, 0.0, 0.0], dtype=float),
                "minimum": 0.0,
                "maximum": 1.0,
            }
        ]
    )

    assert remove_calls == [("slice-existing", False)]
    assert len(add_calls) == 1
    assert add_calls[0]["kwargs"]["lighting"] is False
    assert add_calls[0]["kwargs"]["render"] is False
    assert viewer._pyvista_slice_actors == ["slice-1"]
    assert draw_calls["count"] == 1

    viewer.set_slice_planes(None)

    assert remove_calls[-1] == ("slice-1", False)
    assert viewer._pyvista_slice_actors == []
    assert draw_calls["count"] == 2
    widget.close()


def test_atomic_structure_viewer_point_overlays_extend_scene(qapp) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(_sample_structure())
    viewer.set_point_overlays(
        [
            {
                "positions": [(8.0, 0.0, 0.0)],
                "colors": ["#ff5500"],
                "radii": [0.3],
                "alpha": 0.9,
            }
        ]
    )

    _positions, _numbers, _symbols, _colors, _radii, extra_points = viewer._scene_geometry()

    assert viewer._point_overlay_specs
    assert any(np.allclose(point, [8.0, 0.0, 0.0]) for point in extra_points)
    viewer.close()


def test_atomic_structure_viewer_updates_pyvista_point_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []
    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"actor-{len(add_calls)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_point_overlay_actors = ["point-existing"]

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_atom_glyph_mesh",
        lambda *_args, **_kwargs: "glyphs",
    )
    monkeypatch.setattr(viewer, "_render", lambda: pytest.fail("full render should not run for PyVista point-overlay updates"))

    viewer.set_point_overlays(
        [
            {
                "positions": [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
                "colors": ["#ff5500", "#00aa88"],
                "radii": [0.2, 0.2],
            }
        ]
    )

    assert remove_calls == [("point-existing", False)]
    assert viewer._pyvista_point_overlay_actors == ["actor-1"]
    assert draw_calls["count"] == 1

    viewer.set_point_overlays(None)

    assert remove_calls[-1] == ("actor-1", False)
    assert viewer._pyvista_point_overlay_actors == []
    assert draw_calls["count"] == 2
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_selection_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            return f"selection-{kwargs.get('opacity', 0.0)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_all_atom_positions = viewer._pyvista_atom_positions.copy()
    viewer._pyvista_all_atom_radii = viewer._pyvista_atom_radii.copy()
    viewer._pyvista_selection_actor = "selection-existing"

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(viewer, "_draw_pyvista_selection", lambda *_args: setattr(viewer, "_pyvista_selection_actor", "selection-new"))
    monkeypatch.setattr(viewer, "_render", lambda: pytest.fail("full render should not run for PyVista selection-only updates"))

    viewer.select_atom(1)

    assert remove_calls == [("selection-existing", False)]
    assert viewer._pyvista_selection_actor == "selection-new"
    assert draw_calls["count"] == 1

    viewer.select_atom(None)

    assert remove_calls[-1] == ("selection-new", False)
    assert viewer._pyvista_selection_actor is None
    assert draw_calls["count"] == 2
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_multi_selection_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            return f"selection-{kwargs.get('opacity', 0.0)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_all_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_all_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_selection_actor = "selection-existing"

    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(viewer, "_draw_pyvista_selection", lambda *_args: setattr(viewer, "_pyvista_selection_actor", "selection-new"))
    monkeypatch.setattr(viewer, "_render", lambda: pytest.fail("full render should not run for PyVista selection-only updates"))

    viewer.select_atoms([0, 1])

    assert remove_calls == [("selection-existing", False)]
    assert viewer._pyvista_selection_actor == "selection-new"
    assert draw_calls["count"] == 1
    widget.close()


def test_atomic_structure_viewer_hides_atoms_in_front_of_all_slice_planes(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer
    positions = np.asarray(viewer._atoms.get_positions(), dtype=float)

    _set_slice_planes_hiding_second_atom(viewer)

    hidden = viewer._hidden_atom_indexes_for_slices(positions)

    assert hidden.tolist() == [False, True]
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_bond_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._atoms = viewer._to_atoms(_sample_structure())
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_colors = np.array([[1.0, 0.0, 0.0], [0.2, 0.8, 0.1]], dtype=float)
    viewer._pyvista_bond_actors = ["bond-existing"]

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(viewer, "_draw_pyvista_bonds", lambda *_args, **_kwargs: ["bond-new"])

    viewer._show_bonds.setChecked(True)
    assert viewer._try_update_pyvista_bond_overlay() is True
    assert remove_calls[0] == ("bond-existing", False)
    assert viewer._pyvista_bond_actors == ["bond-new"]
    assert draw_calls["count"] >= 1

    viewer._show_bonds.setChecked(False)
    viewer._try_update_pyvista_bond_overlay()
    assert remove_calls[-1] == ("bond-new", False)
    assert viewer._pyvista_bond_actors == []
    assert draw_calls["count"] >= 2
    widget.close()


def test_atomic_structure_viewer_pyvista_overlays_hide_slice_culled_bonds_and_spins(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer
    _set_slice_planes_hiding_second_atom(viewer)

    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}
    bond_capture: dict[str, np.ndarray] = {}
    spin_capture: dict[str, np.ndarray] = {}

    class _FakePlotter:
        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    positions, _numbers, _symbols, colors, radii, _extra_points = viewer._scene_geometry()
    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._atoms = viewer._to_atoms(_sample_structure())
    viewer._pyvista_all_atom_positions = positions.copy()
    viewer._pyvista_all_atom_radii = radii.copy()
    viewer._pyvista_all_atom_colors = colors.copy()
    viewer._spins = np.array([1.0, -1.5], dtype=float)

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))

    def _capture_bonds(_plotter, _pv, positions, colors, radii, *, visible_mask=None, alpha):
        del positions, colors, radii, alpha
        assert visible_mask is not None
        bond_capture["visible_mask"] = np.asarray(visible_mask, dtype=bool).copy()
        return ["bond-new"]

    def _capture_spins(_plotter, _pv, positions, radii, spins):
        del radii
        spin_capture["positions"] = np.asarray(positions, dtype=float).copy()
        spin_capture["spins"] = np.asarray(spins, dtype=float).copy()
        return ["spin-new"]

    monkeypatch.setattr(viewer, "_draw_pyvista_bonds", _capture_bonds)
    monkeypatch.setattr(viewer, "_draw_pyvista_spins", _capture_spins)

    viewer._show_bonds.setChecked(True)
    assert viewer._try_update_pyvista_bond_overlay() is True
    assert bond_capture["visible_mask"].tolist() == [True, False]

    viewer._show_spins.setChecked(True)
    assert viewer._try_update_pyvista_spin_overlay() is True
    assert spin_capture["positions"].shape == (1, 3)
    assert np.allclose(spin_capture["spins"], [1.0])
    assert draw_calls["count"] >= 2
    widget.close()


def test_atomic_structure_viewer_updates_pyvista_spin_overlay_without_full_rerender(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._spins = np.array([1.0, -1.5], dtype=float)
    viewer._pyvista_spin_actors = ["spin-existing"]

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(viewer, "_draw_pyvista_spins", lambda *_args, **_kwargs: ["spin-new"])

    viewer._show_spins.setChecked(True)
    assert viewer._try_update_pyvista_spin_overlay() is True
    assert remove_calls[0] == ("spin-existing", False)
    assert viewer._pyvista_spin_actors == ["spin-new"]
    assert draw_calls["count"] >= 1

    viewer._show_spins.setChecked(False)
    assert viewer._try_update_pyvista_spin_overlay() is True
    assert remove_calls[-1] == ("spin-new", False)
    assert viewer._pyvista_spin_actors == []
    assert draw_calls["count"] >= 2
    widget.close()


def test_atomic_structure_viewer_hides_pyvista_spins_without_full_rerender_when_atom_opacity_updates(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    class _FakeProperty:
        def __init__(self) -> None:
            self.opacity = None

        def SetOpacity(self, value: float) -> None:
            self.opacity = value

    class _FakeActor:
        def __init__(self) -> None:
            self.prop = _FakeProperty()

        def GetProperty(self):
            return self.prop

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(plotter=SimpleNamespace())
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._spins = np.array([1.0, -1.5], dtype=float)
    viewer._pyvista_spin_actors = []
    viewer._pyvista_atom_actor = _FakeActor()

    draw_calls = {"count": 0}
    viewer._canvas.draw_idle = lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1)
    viewer._show_spins.setChecked(False)

    monkeypatch.setattr(viewer, "_remove_pyvista_actor_group", lambda _actors: False)

    assert viewer._try_update_pyvista_spin_overlay() is True
    assert viewer._pyvista_atom_actor.prop.opacity == pytest.approx(1.0)
    assert draw_calls["count"] >= 1
    widget.close()


def test_atomic_structure_viewer_pyvista_camera_poll_redraws_triad_canvas(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    viewer = widget._viewer
    calls = {"triad": 0}
    camera = SimpleNamespace(
        position=(9.0, 8.0, 7.0),
        focal_point=(1.0, 2.0, 3.0),
        up=(0.0, 1.0, 0.0),
        parallel_scale=2.5,
    )
    viewer._canvas = SimpleNamespace(plotter=SimpleNamespace(camera=camera))
    viewer._pyvista_last_camera_signature = None
    monkeypatch.setattr(viewer, "_draw_crystallographic_axes", lambda: calls.__setitem__("triad", calls["triad"] + 1))

    viewer._poll_pyvista_camera()

    assert calls["triad"] == 1
    widget.close()


def test_build_pyvista_sphere_mesh_returns_centered_sphere_mesh() -> None:
    pv = pytest.importorskip("pyvista")

    mesh = _build_pyvista_sphere_mesh(pv, radius=1.2, center=(1.0, 2.0, 3.0), subdivisions=2)

    assert mesh.n_points > 0
    assert mesh.n_cells > 0
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    assert np.allclose(np.mean(bounds, axis=1), [1.0, 2.0, 3.0], atol=0.02)


def test_build_pyvista_atom_glyph_mesh_carries_color_and_scale_arrays() -> None:
    pv = pytest.importorskip("pyvista")

    glyphs = _build_pyvista_atom_glyph_mesh(
        pv,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        radii=np.array([0.5, 0.8], dtype=float),
        colors=np.array([[1.0, 0.0, 0.0], [0.2, 0.8, 0.1]], dtype=float),
        subdivisions=3,
    )

    assert glyphs.n_points > 0
    assert glyphs.n_cells > 0
    assert "atom_rgb" in glyphs.array_names
    assert glyphs["atom_rgb"].shape[1] == 3
    assert "glyph_radius" in glyphs.array_names


def test_build_pyvista_bond_line_mesh_carries_rgb_array() -> None:
    pv = pytest.importorskip("pyvista")

    line_mesh = _build_pyvista_bond_line_mesh(
        pv,
        segments=[
            (
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
            ),
            (
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
            ),
        ],
    )

    assert line_mesh is not None
    assert line_mesh.n_points == 4
    assert "rgb" in line_mesh.array_names
    assert line_mesh["rgb"].shape == (2, 3)


def test_build_pyvista_bond_tube_mesh_carries_color_array() -> None:
    pv = pytest.importorskip("pyvista")

    bond_mesh = _build_pyvista_bond_tube_mesh(
        pv,
        segments=[
            (
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                0.1,
            ),
            (
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                0.2,
            ),
        ],
    )

    assert bond_mesh is not None
    assert bond_mesh.n_points > 0
    assert bond_mesh.n_cells > 0
    assert "rgb" in bond_mesh.array_names
    assert bond_mesh["rgb"].shape[1] == 3


def test_draw_atom_spheres_plots_one_surface_per_atom() -> None:
    class _FakeAxis:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plot_surface(self, x, y, z, **kwargs):
            self.calls.append({"x": x, "y": y, "z": z, "kwargs": kwargs})
            return f"surface-{len(self.calls)}"

    axis = _FakeAxis()
    artists = _draw_atom_spheres(
        axis,
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.5, -0.5]], dtype=float),
        radii=np.array([0.5, 0.25], dtype=float),
        colors=np.array([[1.0, 0.0, 0.0], [0.2, 0.4, 0.8]], dtype=float),
        alpha=0.42,
        resolution=5,
    )

    assert artists == ["surface-1", "surface-2"]
    assert len(axis.calls) == 2
    assert axis.calls[0]["x"].shape == (5, 5)
    assert axis.calls[1]["kwargs"]["alpha"] == pytest.approx(0.42)


def test_viewer_background_helpers_and_color_blending_are_stable(qapp) -> None:
    assert _blend_hex_colors("#000000", "#ffffff", 0.5) == "#808080"

    style = _viewer_style()
    assert style.background == _viewer_background_color()
    assert len(style.triad_axis_colors) == 3

    captured: dict[str, str] = {}

    class _GoodPlotter:
        def set_background(self, color: str) -> None:
            captured["color"] = color

    class _BadPlotter:
        def set_background(self, _color: str) -> None:
            raise RuntimeError("boom")

    _configure_transparent_pyvista_background(_GoodPlotter())
    _configure_transparent_pyvista_background(_BadPlotter())

    assert captured["color"] == style.background


def test_shutdown_pyvista_widget_stops_render_timer_and_closes_plotter() -> None:
    events: list[str] = []

    class _Timer:
        def stop(self) -> None:
            events.append("stop")

    class _Plotter:
        def __init__(self) -> None:
            self.render_timer = _Timer()

        def close(self) -> None:
            events.append("close")

    _shutdown_pyvista_widget(None)
    _shutdown_pyvista_widget(SimpleNamespace(plotter=_Plotter()))

    assert events == ["stop", "close"]


def test_shutdown_pyvista_widget_ignores_timer_and_close_errors() -> None:
    events: list[str] = []

    class _Timer:
        def stop(self) -> None:
            events.append("stop")
            raise RuntimeError("timer")

    class _Plotter:
        def __init__(self) -> None:
            self.render_timer = _Timer()

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close")

    _shutdown_pyvista_widget(SimpleNamespace(_plotter=_Plotter()))

    assert events == ["stop", "close"]


def test_plot_cylinder_segment_handles_degenerate_and_vertical_segments() -> None:
    class _FakeAxis:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plot_surface(self, x, y, z, **kwargs) -> None:
            self.calls.append({"x": x, "y": y, "z": z, "kwargs": kwargs})

    axis = _FakeAxis()
    assert _plot_cylinder_segment(
        axis,
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.0], dtype=float),
        radius=0.15,
        color=np.array([1.0, 0.0, 0.0], dtype=float),
        alpha=0.3,
        resolution=6,
    ) is False

    assert _plot_cylinder_segment(
        axis,
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 2.0], dtype=float),
        radius=0.2,
        color=np.array([0.1, 0.4, 0.8], dtype=float),
        alpha=0.55,
        resolution=6,
    ) is True
    assert len(axis.calls) == 1
    assert axis.calls[0]["x"].shape == (6, 2)
    assert axis.calls[0]["kwargs"]["alpha"] == pytest.approx(0.55)


def test_atomic_structure_viewer_draws_pyvista_unit_cell_edges(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"edge-{len(add_calls)}"

    class _FakePv:
        @staticmethod
        def Line(start, end):
            return ("line", tuple(np.asarray(start, dtype=float)), tuple(np.asarray(end, dtype=float)))

    viewer._draw_pyvista_unit_cell(_FakePlotter(), _FakePv)

    assert len(add_calls) == 12
    assert all(call["kwargs"]["opacity"] == pytest.approx(0.62) for call in add_calls)
    widget.close()


def test_atomic_structure_viewer_draws_pyvista_isosurfaces_and_tracks_primary_actor(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"iso-{len(add_calls)}"

    monkeypatch.setattr(
        viewer,
        "_build_pyvista_isosurface_mesh",
        lambda _pv, *, vertices, faces: {"vertices": np.asarray(vertices), "faces": np.asarray(faces)},
    )
    viewer._isosurface_specs = [
        _ViewerIsosurfaceSpec(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=float),
            faces=np.empty((0, 3), dtype=int),
            color="#000000",
            alpha=0.1,
        ),
        _ViewerIsosurfaceSpec(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=int),
            color="#55aaee",
            alpha=0.35,
        ),
    ]

    viewer._draw_pyvista_isosurfaces(_FakePlotter(), object())

    assert len(add_calls) == 1
    assert add_calls[0]["kwargs"]["name"] == "isosurface-overlay-1"
    assert viewer._pyvista_isosurface_actor == "iso-1"
    assert viewer._pyvista_isosurface_actors == ["iso-1"]
    widget.close()


def test_atomic_structure_viewer_draws_pyvista_slices_and_point_overlays(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"actor-{len(add_calls)}"

    class _FakeStructuredGrid(dict):
        pass

    class _FakePv:
        @staticmethod
        def StructuredGrid(x_coordinates, y_coordinates, z_coordinates):
            return _FakeStructuredGrid(
                x_coordinates=np.asarray(x_coordinates),
                y_coordinates=np.asarray(y_coordinates),
                z_coordinates=np.asarray(z_coordinates),
            )

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_atom_glyph_mesh",
        lambda *_args, **_kwargs: "glyphs",
    )

    viewer._slice_specs = [
        _ViewerSliceSpec(
            x_coordinates=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
            y_coordinates=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            z_coordinates=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            anchor=np.array([0.0, 0.0, 0.0], dtype=float),
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
            color_limits=(1.0, 4.0),
            cmap="viridis",
            alpha=0.4,
        )
    ]
    viewer._point_overlay_specs = [
        _ViewerPointOverlaySpec(
            positions=np.empty((0, 3), dtype=float),
            colors=np.empty((0, 3), dtype=float),
            radii=np.empty((0,), dtype=float),
            alpha=0.2,
        ),
        _ViewerPointOverlaySpec(
            positions=np.array([[1.0, 2.0, 3.0]], dtype=float),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=float),
            radii=np.array([0.25], dtype=float),
            alpha=0.8,
        ),
    ]

    plotter = _FakePlotter()
    viewer._draw_pyvista_slices(plotter, _FakePv)
    viewer._draw_pyvista_point_overlays(plotter, object())

    assert viewer._pyvista_slice_actors == ["actor-1"]
    assert np.allclose(add_calls[0]["mesh"]["slice_value"], [1.0, 3.0, 2.0, 4.0])
    assert add_calls[0]["kwargs"]["name"] == "slice-overlay-0"
    assert viewer._pyvista_point_overlay_actors == ["actor-2"]
    assert add_calls[1]["mesh"] == "glyphs"
    assert add_calls[1]["kwargs"]["name"] == "point-overlay-1"
    widget.close()


def test_atomic_structure_viewer_draws_selection_overlay_and_routes_visibility_updates(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakeSphere:
        def __init__(self, label: str) -> None:
            self.label = label

        def merge(self, other, *, inplace: bool = False):
            assert inplace is False
            return _FakeSphere(f"{self.label}+{other.label}")

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return "selection-actor"

    spheres = iter([_FakeSphere("first"), _FakeSphere("second")])
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_sphere_mesh",
        lambda *_args, **_kwargs: next(spheres),
    )

    viewer._selected_indexes = [0, 5, 1]
    viewer._draw_pyvista_selection(
        _FakePlotter(),
        object(),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        radii=np.array([0.6, 0.45], dtype=float),
    )

    assert viewer._pyvista_selection_actor == "selection-actor"
    assert add_calls[0]["mesh"].label == "first+second"

    bond_calls = {"count": 0}
    spin_calls = {"count": 0}
    monkeypatch.setattr(viewer, "_try_update_pyvista_bond_overlay", lambda: bond_calls.__setitem__("count", bond_calls["count"] + 1) or True)
    monkeypatch.setattr(viewer, "_try_update_pyvista_spin_overlay", lambda: spin_calls.__setitem__("count", spin_calls["count"] + 1) or True)

    viewer._backend_name = "pyvista"
    viewer._shutting_down = False
    monkeypatch.setattr(viewer, "sender", lambda: viewer._show_bonds)
    assert viewer._try_update_pyvista_visibility_overlay() is True
    monkeypatch.setattr(viewer, "sender", lambda: viewer._show_spins)
    assert viewer._try_update_pyvista_visibility_overlay() is True
    monkeypatch.setattr(viewer, "sender", lambda: None)
    assert viewer._try_update_pyvista_visibility_overlay() is False

    assert bond_calls["count"] == 1
    assert spin_calls["count"] == 1
    widget.close()


def test_atomic_structure_viewer_updates_isosurface_overlay_branches(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []
    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"iso-{len(add_calls)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._atoms = viewer._to_atoms(_sample_structure())
    viewer._pyvista_isosurface_actor = "iso-existing"
    viewer._pyvista_isosurface_actors = ["iso-existing"]
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None))
    monkeypatch.setattr(
        viewer,
        "_build_pyvista_isosurface_mesh",
        lambda _pv, *, vertices, faces: {"vertices": np.asarray(vertices), "faces": np.asarray(faces)},
    )

    viewer._isosurface_specs = []
    assert viewer._try_update_pyvista_isosurface_overlay() is True
    assert remove_calls == [("iso-existing", False)]
    assert draw_calls["count"] == 1

    viewer._isosurface_specs = [
        _ViewerIsosurfaceSpec(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=float),
            faces=np.empty((0, 3), dtype=int),
            color="#000000",
            alpha=0.1,
        ),
        _ViewerIsosurfaceSpec(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=int),
            color="#ffaa33",
            alpha=0.5,
        ),
    ]
    assert viewer._try_update_pyvista_isosurface_overlay() is True
    assert add_calls[0]["kwargs"]["name"] == "isosurface-overlay-1"
    assert viewer._pyvista_isosurface_actor == "iso-1"
    assert viewer._pyvista_isosurface_actors == ["iso-1"]
    assert draw_calls["count"] == 2
    widget.close()


def test_atomic_structure_viewer_updates_slice_and_point_overlay_branches(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []
    remove_calls: list[tuple[object, bool]] = []
    draw_calls = {"count": 0}

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"actor-{len(add_calls)}"

        def remove_actor(self, actor, *, render: bool = True) -> None:
            remove_calls.append((actor, render))

    class _FakeStructuredGrid(dict):
        pass

    class _FakePv:
        @staticmethod
        def StructuredGrid(x_coordinates, y_coordinates, z_coordinates):
            return _FakeStructuredGrid(
                x_coordinates=np.asarray(x_coordinates),
                y_coordinates=np.asarray(y_coordinates),
                z_coordinates=np.asarray(z_coordinates),
            )

    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(
        plotter=_FakePlotter(),
        draw_idle=lambda: draw_calls.__setitem__("count", draw_calls["count"] + 1),
    )
    viewer._atoms = viewer._to_atoms(_sample_structure())
    viewer._pyvista_slice_actors = ["slice-existing"]
    viewer._pyvista_point_overlay_actors = ["point-existing"]

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (_FakePv, None))
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_atom_glyph_mesh",
        lambda *_args, **_kwargs: "glyphs",
    )

    viewer._slice_specs = []
    assert viewer._try_update_pyvista_slice_overlay() is True
    viewer._point_overlay_specs = []
    assert viewer._try_update_pyvista_point_overlay() is True
    assert remove_calls[:2] == [("slice-existing", False), ("point-existing", False)]

    viewer._slice_specs = [
        _ViewerSliceSpec(
            x_coordinates=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
            y_coordinates=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            z_coordinates=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            anchor=np.array([0.0, 0.0, 0.0], dtype=float),
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
            color_limits=(1.0, 4.0),
            cmap="viridis",
            alpha=0.4,
        )
    ]
    viewer._point_overlay_specs = [
        _ViewerPointOverlaySpec(
            positions=np.array([[1.0, 2.0, 3.0]], dtype=float),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=float),
            radii=np.array([0.25], dtype=float),
            alpha=0.8,
        )
    ]
    assert viewer._try_update_pyvista_slice_overlay() is True
    assert viewer._try_update_pyvista_point_overlay() is True
    assert viewer._pyvista_slice_actors == ["actor-1"]
    assert viewer._pyvista_point_overlay_actors == ["actor-2"]
    assert add_calls[0]["kwargs"]["name"] == "slice-overlay-0"
    assert add_calls[1]["kwargs"]["name"] == "point-overlay-0"
    assert draw_calls["count"] == 4
    widget.close()


def test_atomic_structure_viewer_draw_pyvista_bonds_uses_fast_tube_mesh(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return "tube-actor"

    bonded_structure = Structure(
        label="Bonded",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
    )
    viewer._atoms = viewer._to_atoms(bonded_structure)
    positions = np.asarray(viewer._atoms.get_positions(), dtype=float)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    radii = np.array([0.6, 0.45], dtype=float)

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_cylinders", lambda *_args: True)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_lines", lambda *_args: False)
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._trim_bond_segment_to_surfaces",
        lambda start, end, **_kwargs: (np.asarray(start, dtype=float), np.asarray(end, dtype=float)),
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_bond_tube_mesh",
        lambda _pv, *, segments: {"segments": list(segments)},
    )

    actors = viewer._draw_pyvista_bonds(_FakePlotter(), object(), positions, colors, radii, alpha=0.6)

    assert actors == ["tube-actor"]
    assert len(add_calls[0]["mesh"]["segments"]) == 1
    assert add_calls[0]["kwargs"]["render"] is False
    widget.close()


def test_atomic_structure_viewer_draw_pyvista_bonds_uses_fast_line_mesh(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return "line-actor"

    bonded_structure = Structure(
        label="Bonded",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
    )
    viewer._atoms = viewer._to_atoms(bonded_structure)
    positions = np.asarray(viewer._atoms.get_positions(), dtype=float)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    radii = np.array([0.6, 0.45], dtype=float)

    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_cylinders", lambda *_args: False)
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_lines", lambda *_args: True)
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_bond_line_mesh",
        lambda _pv, *, segments: {"segments": list(segments)},
    )

    actors = viewer._draw_pyvista_bonds(_FakePlotter(), object(), positions, colors, radii, alpha=0.6)

    assert actors == ["line-actor"]
    assert len(add_calls[0]["mesh"]["segments"]) == 2
    assert add_calls[0]["kwargs"]["lighting"] is False
    widget.close()


def test_atomic_structure_viewer_normalizes_pyvista_light_direction(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())

    widget._viewer.set_pyvista_light_direction((3.0, 0.0, 0.0))

    assert np.allclose(widget._viewer._pyvista_light_direction, [1.0, 0.0, 0.0])
    widget.close()


def test_atomic_structure_viewer_reuses_pyvista_camera_snapshot_on_redraw(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    camera = SimpleNamespace(
        position=(9.0, 8.0, 7.0),
        focal_point=(1.0, 2.0, 3.0),
        up=(0.0, 1.0, 0.0),
        parallel_scale=2.5,
    )
    plotter = SimpleNamespace(camera=camera, reset_camera_clipping_range=lambda: None)
    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(plotter=plotter)
    viewer._limits = ((0.0, 4.0), (0.0, 5.0), (0.0, 6.0))

    monkeypatch.setattr(viewer, "_configure_pyvista_lighting", lambda *_args, **_kwargs: None)

    viewer._capture_pyvista_camera_snapshot()
    camera.position = (0.0, 0.0, 0.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 0.0, 1.0)
    camera.parallel_scale = 0.25

    viewer._apply_pyvista_camera()

    assert camera.position == (9.0, 8.0, 7.0)
    assert camera.focal_point == (1.0, 2.0, 3.0)
    assert camera.up == (0.0, 1.0, 0.0)
    assert camera.parallel_scale == pytest.approx(2.5)
    widget.close()


def test_atomic_structure_viewer_set_structure_preserving_view_keeps_camera_limits_and_spin_toggle(qapp) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(_sample_structure())
    viewer._camera_state = _CameraState(17.0, 38.0, 9.0)
    viewer._axis.set_from_state(viewer._camera_state)
    viewer._view = (17.0, 38.0)
    viewer._view_roll = 9.0
    viewer._limits = ((0.0, 4.0), (0.0, 5.0), (0.0, 6.0))
    viewer._show_spins.setChecked(False)
    viewer.select_atoms([1])

    updated = Structure(
        label="Host shifted",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        species=["Si", "O"],
        fractional_positions=[[0.05, 0.0, 0.0], [0.30, 0.5, 0.75]],
        properties={"pbc": [True, True, True], "atomic_spins": [1.0, -1.5]},
    )

    viewer.set_structure_preserving_view(updated)

    assert viewer._camera_state == _CameraState(17.0, 38.0, 9.0)
    assert viewer._limits == ((0.0, 4.0), (0.0, 5.0), (0.0, 6.0))
    assert not viewer._show_spins.isChecked()
    assert viewer.selected_atom_indices() == (1,)
    viewer.close()


def test_atomic_structure_viewer_update_structure_geometry_preserving_view_keeps_camera_and_updates_positions(qapp) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(_sample_structure())
    viewer._camera_state = _CameraState(19.0, 44.0, 7.0)
    viewer._axis.set_from_state(viewer._camera_state)
    viewer._view = (19.0, 44.0)
    viewer._view_roll = 7.0
    viewer._limits = ((0.0, 4.0), (0.0, 5.0), (0.0, 6.0))
    viewer.select_atoms([1])

    updated = Structure(
        label="Host shifted",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        species=["Si", "O"],
        fractional_positions=[[0.10, 0.0, 0.0], [0.35, 0.5, 0.75]],
        properties={"pbc": [True, True, True], "atomic_spins": [1.0, -1.5]},
    )

    viewer.update_structure_geometry_preserving_view(updated)

    assert viewer._camera_state == _CameraState(19.0, 44.0, 7.0)
    assert viewer._limits == ((0.0, 4.0), (0.0, 5.0), (0.0, 6.0))
    assert viewer.selected_atom_indices() == (1,)
    assert viewer._atoms is not None
    assert np.allclose(viewer._atoms.get_scaled_positions(wrap=True), np.asarray(updated.fractional_positions, dtype=float))
    viewer.close()


def test_atomic_structure_viewer_pyvista_zoom_only_poll_updates_camera_snapshot(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    camera = SimpleNamespace(
        position=(9.0, 8.0, 7.0),
        focal_point=(1.0, 2.0, 3.0),
        up=(0.0, 1.0, 0.0),
        parallel_scale=2.5,
    )
    viewer._backend_name = "pyvista"
    viewer._canvas = SimpleNamespace(plotter=SimpleNamespace(camera=camera))
    monkeypatch.setattr(viewer, "_draw_crystallographic_axes", lambda: None)

    viewer._capture_pyvista_camera_snapshot()
    viewer._pyvista_last_camera_signature = viewer._pyvista_camera_signature()
    camera.parallel_scale = 0.75

    viewer._poll_pyvista_camera()

    assert viewer._pyvista_camera_snapshot is not None
    assert viewer._pyvista_camera_snapshot[3] == pytest.approx(0.75)
    widget.close()


def test_atomic_structure_viewer_style_tracks_theme_palette() -> None:
    palette = current_palette()
    style = _viewer_style()

    assert style.background == palette.bg_primary
    assert style.text_primary == palette.text_primary
    assert style.triad_axis_colors == (
        palette.accent,
        palette.success,
        palette.status_submitting,
    )
    assert style.isosurface == palette.status_submitting
    assert style.selection_stroke == palette.accent_pressed
    assert style.triad_label_font_family == body_font_family()


def test_viewer_style_signature_changes_with_active_theme(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    light_signature = _viewer_style_signature()

    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        dark_signature = _viewer_style_signature()

        assert dark_signature != light_signature
        assert theme_module.current_palette().bg_primary in dark_signature
        assert theme_module.current_palette().text_primary in dark_signature
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_atomic_structure_viewer_theme_refresh_rerenders_scene(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = AtomicStructureViewer()
    calls = {"store": 0, "render": 0, "canvas": 0, "triad": 0}

    class _Canvas:
        def refresh_theme(self) -> None:
            calls["canvas"] += 1

    class _Triad:
        def refresh_theme(self) -> None:
            calls["triad"] += 1

    viewer._canvas = _Canvas()
    viewer._triad_canvas = _Triad()
    monkeypatch.setattr(viewer, "_store_view", lambda: calls.__setitem__("store", calls["store"] + 1))
    monkeypatch.setattr(viewer, "_render", lambda: calls.__setitem__("render", calls["render"] + 1))

    viewer.refresh_theme()

    assert calls == {"store": 1, "render": 1, "canvas": 1, "triad": 1}
    viewer.close()


def test_atomic_structure_viewer_coalesces_theme_change_events(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = AtomicStructureViewer()
    calls = {"refresh": 0}
    monkeypatch.setattr(viewer, "refresh_theme", lambda: calls.__setitem__("refresh", calls["refresh"] + 1))

    viewer._style_signature = ("stale",)
    event = QEvent(QEvent.Type.ApplicationPaletteChange)
    viewer.changeEvent(event)
    viewer.changeEvent(event)
    qapp.processEvents()

    assert calls["refresh"] == 1

    viewer._style_signature = _viewer_style_signature()
    viewer.changeEvent(event)
    qapp.processEvents()

    assert calls["refresh"] == 1
    viewer.close()


def test_atomic_structure_viewer_shutdown_closes_pyvista_resources(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(_sample_structure())
    viewer = widget._viewer

    class _FakeTimer:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class _FakePlotter:
        def __init__(self) -> None:
            self.render_timer = _FakeTimer()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    main_plotter = _FakePlotter()
    triad_plotter = _FakePlotter()
    viewer._backend_name = "pyvista"
    viewer._pyvista_camera_timer = _FakeTimer()
    viewer._canvas = SimpleNamespace(plotter=main_plotter)
    viewer._triad_canvas = SimpleNamespace(plotter=triad_plotter)

    viewer.shutdown()
    viewer.shutdown()

    assert viewer._shutting_down is True
    assert viewer._pyvista_camera_timer.stop_calls == 1
    assert main_plotter.render_timer.stop_calls == 1
    assert triad_plotter.render_timer.stop_calls == 1
    assert main_plotter.close_calls == 1
    assert triad_plotter.close_calls == 1
    widget.close()


def test_wrapping_hint_label_reports_height_for_width(qapp) -> None:
    label = _WrappingHintLabel(
        "Left click atom: select | Left click empty: clear | Right drag: rotate | Middle drag: pan | Wheel: zoom"
    )
    label.setWordWrap(True)
    label.resize(220, 10)
    label.show()
    qapp.processEvents()

    assert label.hasHeightForWidth()
    assert label.heightForWidth(220) > label.fontMetrics().lineSpacing()
    assert label.minimumHeight() >= label.heightForWidth(220)
    label.close()


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
