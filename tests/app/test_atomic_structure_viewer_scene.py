"""The structure viewer's scene: meshes, bonds, spins and in-place overlay updates.

Covers the PyVista mesh builders, the drawing helpers, and the isosurface, slice,
point, selection, bond and spin overlays that update without a full re-render.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from mufinder.app.atomic_structure_viewer import (
    AtomicStructureViewer,
    _build_pyvista_atom_glyph_mesh,
    _build_pyvista_bond_line_mesh,
    _build_pyvista_bond_tube_mesh,
    _build_pyvista_sphere_mesh,
    _draw_atom_spheres,
    _draw_spin_arrows,
    _draw_split_bond_cylinder,
    _plot_cylinder_segment,
    _set_equal_limits,
    _ViewerIsosurfaceSpec,
    _ViewerPointOverlaySpec,
    _ViewerSliceSpec,
)
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.domain.models import Structure
from tests.support.structures import magnetic_silicon_oxide_host


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


def test_bond_alpha_tracks_atom_transparency_state(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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


def test_atomic_structure_viewer_renders_isosurface_overlay(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
    monkeypatch.setattr(
        viewer,
        "_render",
        lambda: pytest.fail("full render should not run for PyVista isosurface-only updates"),
    )

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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (_FakePv, None)
    )
    monkeypatch.setattr(
        viewer,
        "_render",
        lambda: pytest.fail("full render should not run for PyVista slice-only updates"),
    )

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
    viewer.set_structure(magnetic_silicon_oxide_host())
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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_atom_glyph_mesh",
        lambda *_args, **_kwargs: "glyphs",
    )
    monkeypatch.setattr(
        viewer,
        "_render",
        lambda: pytest.fail("full render should not run for PyVista point-overlay updates"),
    )

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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
    monkeypatch.setattr(
        viewer,
        "_draw_pyvista_selection",
        lambda *_args: setattr(viewer, "_pyvista_selection_actor", "selection-new"),
    )
    monkeypatch.setattr(
        viewer,
        "_render",
        lambda: pytest.fail("full render should not run for PyVista selection-only updates"),
    )

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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
    monkeypatch.setattr(
        viewer,
        "_draw_pyvista_selection",
        lambda *_args: setattr(viewer, "_pyvista_selection_actor", "selection-new"),
    )
    monkeypatch.setattr(
        viewer,
        "_render",
        lambda: pytest.fail("full render should not run for PyVista selection-only updates"),
    )

    viewer.select_atoms([0, 1])

    assert remove_calls == [("selection-existing", False)]
    assert viewer._pyvista_selection_actor == "selection-new"
    assert draw_calls["count"] == 1
    widget.close()


def test_atomic_structure_viewer_hides_atoms_in_front_of_all_slice_planes(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    viewer._atoms = viewer._to_atoms(magnetic_silicon_oxide_host())
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_colors = np.array([[1.0, 0.0, 0.0], [0.2, 0.8, 0.1]], dtype=float)
    viewer._pyvista_bond_actors = ["bond-existing"]

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    viewer._atoms = viewer._to_atoms(magnetic_silicon_oxide_host())
    viewer._pyvista_all_atom_positions = positions.copy()
    viewer._pyvista_all_atom_radii = radii.copy()
    viewer._pyvista_all_atom_colors = colors.copy()
    viewer._spins = np.array([1.0, -1.5], dtype=float)

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )

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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
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
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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


def test_plot_cylinder_segment_handles_degenerate_and_vertical_segments() -> None:
    class _FakeAxis:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plot_surface(self, x, y, z, **kwargs) -> None:
            self.calls.append({"x": x, "y": y, "z": z, "kwargs": kwargs})

    axis = _FakeAxis()
    assert (
        _plot_cylinder_segment(
            axis,
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
            radius=0.15,
            color=np.array([1.0, 0.0, 0.0], dtype=float),
            alpha=0.3,
            resolution=6,
        )
        is False
    )

    assert (
        _plot_cylinder_segment(
            axis,
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 2.0], dtype=float),
            radius=0.2,
            color=np.array([0.1, 0.4, 0.8], dtype=float),
            alpha=0.55,
            resolution=6,
        )
        is True
    )
    assert len(axis.calls) == 1
    assert axis.calls[0]["x"].shape == (6, 2)
    assert axis.calls[0]["kwargs"]["alpha"] == pytest.approx(0.55)


def test_atomic_structure_viewer_draws_pyvista_isosurfaces_and_tracks_primary_actor(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return f"iso-{len(add_calls)}"

    monkeypatch.setattr(
        viewer,
        "_build_pyvista_isosurface_mesh",
        lambda _pv, *, vertices, faces: {
            "vertices": np.asarray(vertices),
            "faces": np.asarray(faces),
        },
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
    widget.set_structure(magnetic_silicon_oxide_host())
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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    add_calls: list[dict[str, object]] = []

    class _FakePlotter:
        def add_mesh(self, mesh, **kwargs):
            add_calls.append({"mesh": mesh, "kwargs": kwargs})
            return "selection-actor"

    import pyvista as pv

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._selected_indexes = [0, 5, 1]
    viewer._draw_pyvista_selection(
        _FakePlotter(), pv, positions=positions, radii=np.array([0.6, 0.45], dtype=float)
    )

    assert viewer._pyvista_selection_actor == "selection-actor"
    assert viewer._pyvista_selection_atoms == (0, 1)
    # One highlight sphere per drawn atom, 1.18 times its radius, in selection order.
    spheres = np.asarray(add_calls[0]["mesh"].points).reshape(2, -1, 3)
    assert np.allclose(spheres.mean(axis=1), positions, atol=1e-6)
    assert np.allclose(np.ptp(spheres[:, :, 2], axis=1), 2 * 1.18 * np.array([0.6, 0.45]))

    bond_calls = {"count": 0}
    spin_calls = {"count": 0}
    monkeypatch.setattr(
        viewer,
        "_try_update_pyvista_bond_overlay",
        lambda: bond_calls.__setitem__("count", bond_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(
        viewer,
        "_try_update_pyvista_spin_overlay",
        lambda: spin_calls.__setitem__("count", spin_calls["count"] + 1) or True,
    )

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


def test_atomic_structure_viewer_updates_isosurface_overlay_branches(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    viewer._atoms = viewer._to_atoms(magnetic_silicon_oxide_host())
    viewer._pyvista_isosurface_actor = "iso-existing"
    viewer._pyvista_isosurface_actors = ["iso-existing"]
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (object(), None)
    )
    monkeypatch.setattr(
        viewer,
        "_build_pyvista_isosurface_mesh",
        lambda _pv, *, vertices, faces: {
            "vertices": np.asarray(vertices),
            "faces": np.asarray(faces),
        },
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


def test_atomic_structure_viewer_updates_slice_and_point_overlay_branches(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    viewer._atoms = viewer._to_atoms(magnetic_silicon_oxide_host())
    viewer._pyvista_slice_actors = ["slice-existing"]
    viewer._pyvista_point_overlay_actors = ["point-existing"]

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._pyvista_modules", lambda: (_FakePv, None)
    )
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


def test_atomic_structure_viewer_draw_pyvista_bonds_uses_fast_tube_mesh(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_cylinders",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_lines",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._trim_bond_segment_to_surfaces",
        lambda start, end, **_kwargs: (
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
        ),
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_bond_tube_mesh",
        lambda _pv, *, segments: {"segments": list(segments)},
    )

    actors = viewer._draw_pyvista_bonds(
        _FakePlotter(), object(), positions, colors, radii, alpha=0.6
    )

    assert actors == ["tube-actor"]
    assert len(add_calls[0]["mesh"]["segments"]) == 1
    assert add_calls[0]["kwargs"]["render"] is False
    widget.close()


def test_atomic_structure_viewer_draw_pyvista_bonds_uses_fast_line_mesh(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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

    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_cylinders",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._should_use_fast_pyvista_bond_lines",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "mufinder.app.atomic_structure_viewer._build_pyvista_bond_line_mesh",
        lambda _pv, *, segments: {"segments": list(segments)},
    )

    actors = viewer._draw_pyvista_bonds(
        _FakePlotter(), object(), positions, colors, radii, alpha=0.6
    )

    assert actors == ["line-actor"]
    assert len(add_calls[0]["mesh"]["segments"]) == 2
    assert add_calls[0]["kwargs"]["lighting"] is False
    widget.close()


def test_atomic_structure_viewer_normalizes_pyvista_light_direction(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    widget._viewer.set_pyvista_light_direction((3.0, 0.0, 0.0))

    assert np.allclose(widget._viewer._pyvista_light_direction, [1.0, 0.0, 0.0])
    widget.close()
