"""The structure viewer: rendering, selection, camera, theme and shutdown.

The viewer draws each caller change with one render. Every ``add_mesh`` in a
render pass passes ``render=False`` (a real PyVista plotter otherwise re-renders
the scene once per mesh), the cell outline is one actor, and ``batch_update()``
folds several viewer changes into one render. The plotter here is the test fake
from ``tests/conftest.py``; a counting stub on its ``render`` stands in for
VTK's ``Render``.

Meshes, bonds, spins and the in-place overlay updates are tested in
``test_atomic_structure_viewer_scene.py``; the controls' layout in narrow panes
in ``test_atomic_structure_viewer_controls.py``.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from mufinder.app import theme as theme_module
from mufinder.app.atomic_structure_viewer import (
    AtomicStructureViewer,
    _blend_hex_colors,
    _camera_state_from_vectors,
    _camera_vectors_from_state,
    _CameraState,
    _configure_transparent_pyvista_background,
    _PyVistaCanvas,
    _shutdown_pyvista_widget,
    _viewer_background_color,
    _viewer_style,
    _viewer_style_signature,
    _WrappingHintLabel,
)
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.theme import body_font_family, current_palette
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService
from tests.support.pyvista_fakes import (
    FakePlotter,
    FakeRenderTimer,
    count_created_interactors,
    count_scene_renders,
    install_fake_picker,
)
from tests.support.structures import (
    hexagonal_magnesium_oxide_host,
    host_structure,
    magnetic_silicon_oxide_host,
)

_magnetic_host = partial(
    host_structure,
    lattice=4.0,
    species=("Fe", "O"),
    properties={"pbc": [True, True, True], "atomic_spins": [1.0, -1.0]},
)


class _RenderCounter:
    """Count ``render`` calls on a viewer's main and triad plotters.

    The plotters exist only once the viewer is first shown, so the count is
    taken on the fake plotter class and attributed by the canvas that owns it.
    """

    def __init__(self, viewer: AtomicStructureViewer, monkeypatch: pytest.MonkeyPatch) -> None:
        self.main = 0
        self.triad = 0
        self.add_mesh_render_flags: list[object] = []
        counter = self
        original_add_mesh = FakePlotter.add_mesh

        def counting_render(plotter: FakePlotter) -> None:
            if plotter.parent() is viewer._canvas:
                counter.main += 1
            elif plotter.parent() is viewer._triad_canvas:
                counter.triad += 1

        def recording_add_mesh(plotter: FakePlotter, mesh, **kwargs):
            if plotter.parent() is viewer._canvas:
                counter.add_mesh_render_flags.append(kwargs.get("render", True))
            return original_add_mesh(plotter, mesh, **kwargs)

        monkeypatch.setattr(FakePlotter, "render", counting_render)
        monkeypatch.setattr(FakePlotter, "add_mesh", recording_add_mesh)

    def reset(self) -> None:
        self.main = 0
        self.triad = 0
        self.add_mesh_render_flags = []


@pytest.fixture
def viewer(qapp):
    viewer = AtomicStructureViewer()
    viewer.show()
    yield viewer
    viewer.close()


def _point_overlay() -> list[dict[str, object]]:
    return [{"positions": [[1.0, 1.0, 1.0]], "colors": "#e11d48", "radii": [0.3]}]


def _line_overlay() -> list[dict[str, object]]:
    return [{"start": [0.0, 0.0, 0.0], "end": [1.0, 1.0, 1.0], "arrow": True}]


def test_set_structure_with_spins_in_a_batch_renders_once(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    with viewer.batch_update():
        viewer.set_structure(_magnetic_host())
        viewer.set_spins([2.0, -2.0])
        assert counter.main == 0

    assert counter.main == 1
    assert counter.triad == 1
    assert viewer._show_spins.isChecked()
    assert viewer._pyvista_spin_actors


def _camera(viewer: AtomicStructureViewer) -> tuple[float, ...]:
    camera = viewer._canvas.plotter.camera
    return (*camera.position, *camera.focal_point, float(camera.parallel_scale))


def test_a_batched_new_structure_fits_the_camera_to_the_new_cell(qapp) -> None:
    batched = AtomicStructureViewer()
    batched.show()
    batched.set_structure(_magnetic_host(lattice=4.0))
    direct = AtomicStructureViewer()
    direct.show()

    with batched.batch_update():
        batched.set_structure(_magnetic_host(lattice=12.0))
        batched.set_spins([1.0, -1.0])
        batched.select_atoms([0])
    direct.set_structure(_magnetic_host(lattice=12.0))

    assert _camera(batched) == pytest.approx(_camera(direct))
    batched.close()
    direct.close()


def test_selection_signals_fire_after_the_highlight_is_drawn(viewer) -> None:
    viewer.set_structure(_magnetic_host())
    seen: list[object] = []
    viewer.atom_selected.connect(lambda _index: seen.append(viewer._pyvista_selection_actor))

    viewer._set_selected_atoms([1], emit_signal=True)

    assert len(seen) == 1
    assert seen[0] is not None
    assert viewer._batch_depth == 0


def test_a_render_pass_adds_every_mesh_without_rendering(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)
    viewer._show_bonds.setChecked(True)
    counter.reset()

    with viewer.batch_update():
        viewer.set_structure(_magnetic_host())
        viewer.set_point_overlays(_point_overlay())
        viewer.set_line_overlays(_line_overlay())
        viewer.select_atoms([0])

    assert counter.add_mesh_render_flags
    assert all(flag is False for flag in counter.add_mesh_render_flags)
    assert counter.main == 1


def test_unit_cell_outline_is_one_actor(viewer) -> None:
    viewer.set_structure(_magnetic_host())

    outlines = [
        actor
        for actor in viewer._canvas.plotter._actors
        if getattr(actor, "kwargs", {}).get("name") == "unit-cell-outline"
    ]

    assert len(outlines) == 1
    outline = outlines[0]
    assert outline.kwargs["render"] is False
    assert outline.mesh.n_points == 8
    assert outline.mesh.n_lines == 12
    assert np.allclose(outline.mesh.bounds, (0.0, 4.0, 0.0, 4.0, 0.0, 4.0))


def test_hiding_the_cell_removes_the_outline(viewer) -> None:
    viewer.set_structure(_magnetic_host())

    viewer._show_cell.setChecked(False)

    assert not any(
        getattr(actor, "kwargs", {}).get("name") == "unit-cell-outline"
        for actor in viewer._canvas.plotter._actors
    )


def test_each_setter_on_its_own_renders_once(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    viewer.set_structure(_magnetic_host())
    assert (counter.main, counter.triad) == (1, 1)

    counter.reset()
    viewer.set_spins([0.5, -0.5])
    assert (counter.main, counter.triad) == (1, 1)

    counter.reset()
    viewer._show_cell.setChecked(False)
    assert (counter.main, counter.triad) == (1, 1)


def test_set_structure_preserving_view_renders_once(viewer, monkeypatch) -> None:
    viewer.set_structure(_magnetic_host())
    counter = _RenderCounter(viewer, monkeypatch)

    viewer.set_structure_preserving_view(_magnetic_host(lattice=4.5))

    assert counter.main == 1


def test_batched_in_place_overlay_updates_draw_once(viewer, monkeypatch) -> None:
    viewer.set_structure(_magnetic_host())
    counter = _RenderCounter(viewer, monkeypatch)

    with viewer.batch_update():
        viewer.set_point_overlays(_point_overlay())
        viewer.set_line_overlays(_line_overlay())
        viewer.select_atoms([1])

    assert counter.main == 1
    assert len(viewer._pyvista_point_overlay_actors) == 1
    assert len(viewer._pyvista_line_overlay_actors) == 2
    assert viewer._pyvista_selection_actor is not None


def test_nested_batches_render_once_when_the_outer_batch_ends(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    with viewer.batch_update():
        with viewer.batch_update():
            viewer.set_structure(_magnetic_host())
        assert counter.main == 0
        viewer.set_spins([1.0, 1.0])

    assert counter.main == 1


def test_a_batch_that_changes_nothing_does_not_render(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    with viewer.batch_update():
        pass

    assert (counter.main, counter.triad) == (0, 0)


def test_a_batch_left_by_an_exception_still_renders_and_unwinds(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    with pytest.raises(RuntimeError, match="caller failed"):
        with viewer.batch_update():
            viewer.set_structure(_magnetic_host())
            raise RuntimeError("caller failed")

    assert counter.main == 1
    counter.reset()
    viewer.set_spins([1.0, 1.0])
    assert counter.main == 1


def test_a_batch_on_a_shut_down_viewer_does_not_render(viewer, monkeypatch) -> None:
    counter = _RenderCounter(viewer, monkeypatch)

    with viewer.batch_update():
        viewer.set_structure(_magnetic_host())
        viewer.shutdown()

    assert counter.main == 0


def test_host_widget_loads_a_magnetic_host_with_one_render(qapp, monkeypatch) -> None:
    widget = HostStructureWidget()
    widget.show()
    counter = _RenderCounter(widget._viewer, monkeypatch)

    widget.set_structure(_magnetic_host())

    assert counter.main == 1
    assert widget._viewer._show_spins.isChecked()
    assert widget._viewer._spins is not None
    widget.close()


def test_supercell_editor_preview_shows_structure_and_spins_with_one_render(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "preview.mufinder", "Preview")
    host = _magnetic_host()
    service.add_structure(doc, host)
    assert service.structure_atomic_spins(host) == [1.0, -1.0]
    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    dialog.show()
    counter = _RenderCounter(dialog._viewer, monkeypatch)

    dialog._generate_preview()

    assert counter.main == 1
    assert len(dialog._viewer._atoms) == 4
    assert dialog._viewer._spins is not None
    dialog.close()


@pytest.fixture
def hidden_viewer(qapp):
    viewer = AtomicStructureViewer()
    yield viewer
    viewer.close()


def test_a_hidden_viewer_does_not_render_and_renders_once_when_shown(
    hidden_viewer, monkeypatch
) -> None:
    counter = _RenderCounter(hidden_viewer, monkeypatch)

    hidden_viewer.set_structure(_magnetic_host())
    hidden_viewer.set_spins([2.0, -2.0])
    hidden_viewer.select_atoms([0])
    assert (counter.main, counter.triad) == (0, 0)

    hidden_viewer.show()
    assert counter.main == 1
    assert hidden_viewer._pyvista_atom_actor is not None
    assert hidden_viewer._pyvista_spin_actors
    assert hidden_viewer._pyvista_selection_actor is not None

    counter.reset()
    hidden_viewer.hide()
    hidden_viewer.show()
    assert counter.main == 0


def test_a_change_while_hidden_draws_once_when_shown_again(viewer, monkeypatch) -> None:
    viewer.set_structure(_magnetic_host())
    viewer.hide()
    counter = _RenderCounter(viewer, monkeypatch)

    viewer.set_point_overlays(_point_overlay())
    viewer.set_line_overlays(_line_overlay())
    viewer.set_structure_preserving_view(_magnetic_host(lattice=4.5))
    assert (counter.main, counter.triad) == (0, 0)

    viewer.show()
    assert counter.main == 1
    assert len(viewer._atoms) == 2
    assert viewer._atoms.cell[0][0] == pytest.approx(4.5)


def _shown_viewer_camera(structure) -> tuple[float, ...]:
    direct = AtomicStructureViewer()
    direct.show()
    direct.set_structure(structure)
    camera = _camera(direct)
    direct.close()
    return camera


def test_a_hidden_viewer_refreshes_its_theme_once_when_shown(hidden_viewer, monkeypatch) -> None:
    hidden_viewer.set_structure(_magnetic_host(lattice=7.0))
    hidden_viewer._style_signature = ("stale",)
    counter = _RenderCounter(hidden_viewer, monkeypatch)

    hidden_viewer.refresh_theme()
    assert (counter.main, counter.triad) == (0, 0)
    assert hidden_viewer._style_signature == ("stale",)

    hidden_viewer.show()
    assert counter.main == 1
    assert hidden_viewer._style_signature != ("stale",)
    # The new structure is fitted, not drawn with the never-shown plotter's camera.
    assert _camera(hidden_viewer) == pytest.approx(
        _shown_viewer_camera(_magnetic_host(lattice=7.0))
    )


def test_a_viewer_shown_inside_a_batch_draws_and_refreshes_its_theme_when_the_batch_ends(
    hidden_viewer, monkeypatch
) -> None:
    hidden_viewer._style_signature = ("stale",)
    counter = _RenderCounter(hidden_viewer, monkeypatch)

    with hidden_viewer.batch_update():
        hidden_viewer.set_structure(_magnetic_host(lattice=7.0))
        hidden_viewer.show()
        assert counter.main == 0

    assert counter.main == 1
    assert hidden_viewer._style_signature != ("stale",)
    assert _camera(hidden_viewer) == pytest.approx(
        _shown_viewer_camera(_magnetic_host(lattice=7.0))
    )


def test_a_viewer_creates_its_render_windows_on_first_show(qapp, monkeypatch) -> None:
    interactors = count_created_interactors(monkeypatch)
    viewer = AtomicStructureViewer()
    try:
        viewer.set_structure(_magnetic_host())
        viewer.set_spins([2.0, -2.0])
        viewer.set_point_overlays(_point_overlay())
        assert interactors == []
        assert viewer._canvas.plotter is None

        renders = count_scene_renders(monkeypatch)
        viewer.show()

        # One interactor for the scene and one for the triad, and one scene render.
        assert len(interactors) == 2
        assert {viewer._canvas.plotter, viewer._triad_canvas.plotter} == set(interactors)
        assert viewer._canvas.plotter.isVisible()
        assert renders == {viewer: 1}
        assert viewer._pyvista_atom_actor is not None
        assert viewer._pyvista_spin_actors
        assert viewer._pyvista_point_overlay_actors
        # The triad is built from the cell it was given, in its own interactor.
        assert viewer._triad_canvas.plotter._actors

        viewer.hide()
        viewer.show()
        assert len(interactors) == 2
    finally:
        viewer.close()


def test_a_viewer_closed_before_it_is_shown_creates_no_render_window(qapp, monkeypatch) -> None:
    interactors = count_created_interactors(monkeypatch)
    viewer = AtomicStructureViewer()
    viewer.set_structure(_magnetic_host())
    viewer.close()
    viewer.show()

    assert interactors == []


# --- Same-topology updates move the drawn atoms -------------------------------

# Two bonded Si atoms 2.0 Å apart along x in a 5 Å cube.
_bonded_pair = partial(host_structure, positions=((0.1, 0.2, 0.3), (0.5, 0.2, 0.3)))


def _block_centres(mesh, count: int) -> np.ndarray:
    """The centre of each of ``count`` equal point blocks of a glyph mesh."""

    return np.asarray(mesh.points).reshape(count, -1, 3).mean(axis=1)


def _assert_cylinder_between(actor, start: np.ndarray, end: np.ndarray) -> None:
    """Assert that ``actor`` draws its cylinder, under its transform, from ``start`` to ``end``."""

    points = np.asarray(actor.mesh.points, dtype=float)
    centred = points - points.mean(axis=0)
    built_axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    built_radius = np.linalg.norm(centred - np.outer(centred @ built_axis, built_axis), axis=1)
    drawn = points @ actor.user_matrix[:3, :3].T + actor.user_matrix[:3, 3]
    length = np.linalg.norm(end - start)
    axis = (end - start) / length
    along = (drawn - start) @ axis
    across = np.linalg.norm(drawn - start - np.outer(along, axis), axis=1)
    assert np.allclose([along.min(), along.max()], [0.0, length], atol=1e-5)
    assert np.allclose(across, built_radius, atol=1e-5)


def _scene_actors(viewer: AtomicStructureViewer) -> tuple[object, ...]:
    return (
        viewer._pyvista_atom_actor,
        viewer._pyvista_pick_actor,
        viewer._pyvista_selection_actor,
        *viewer._pyvista_bond_actors,
    )


@pytest.fixture
def bonded_viewer(viewer):
    viewer._show_bonds.setChecked(True)
    viewer.set_structure(_bonded_pair())
    viewer.select_atoms([1])
    return viewer


@pytest.mark.parametrize(
    "update",
    [
        AtomicStructureViewer.set_structure_preserving_view,
        AtomicStructureViewer.update_structure_geometry_preserving_view,
    ],
)
def test_a_same_topology_update_moves_the_drawn_atoms_without_new_meshes(
    bonded_viewer, monkeypatch, update
) -> None:
    viewer = bonded_viewer
    actors = _scene_actors(viewer)
    assert len(viewer._pyvista_bond_actors) == 2
    renders = count_scene_renders(monkeypatch)
    counter = _RenderCounter(viewer, monkeypatch)
    moved = _bonded_pair(positions=((0.12, 0.2, 0.3), (0.54, 0.25, 0.3)))

    update(viewer, moved)

    assert counter.add_mesh_render_flags == []
    assert renders[viewer] == 0
    assert counter.main == 1
    assert _scene_actors(viewer) == actors
    expected = np.asarray(moved.fractional_positions) * 5.0
    assert np.allclose(_block_centres(viewer._pyvista_atom_actor.mesh, 2), expected)
    assert np.allclose(viewer._pyvista_pick_actor.mesh.points, expected)
    assert np.allclose(_block_centres(viewer._pyvista_selection_actor.mesh, 1), expected[1])
    # The bond is drawn as two half cylinders that meet at the new midpoint.
    midpoint = expected.mean(axis=0)
    first_half, second_half = viewer._pyvista_bond_actors
    _assert_cylinder_between(first_half, expected[0], midpoint)
    _assert_cylinder_between(second_half, midpoint, expected[1])
    assert viewer.selected_atom_indices() == (1,)


def test_moving_atoms_keeps_the_camera(bonded_viewer) -> None:
    viewer = bonded_viewer
    viewer._set_named_view("a")
    camera = _camera(viewer)

    viewer.set_structure_preserving_view(
        _bonded_pair(positions=((0.1, 0.2, 0.3), (0.45, 0.2, 0.3)))
    )

    assert _camera(viewer) == camera


def test_a_bond_that_breaks_redraws_only_the_bonds(bonded_viewer, monkeypatch) -> None:
    viewer = bonded_viewer
    atom_actor = viewer._pyvista_atom_actor
    renders = count_scene_renders(monkeypatch)

    viewer.set_structure_preserving_view(_bonded_pair(positions=((0.1, 0.2, 0.3), (0.9, 0.2, 0.3))))

    assert renders[viewer] == 0
    assert viewer._pyvista_atom_actor is atom_actor
    assert viewer._pyvista_bond_actors == []


@pytest.mark.parametrize(
    "changed",
    [
        _bonded_pair(species=("Si", "O")),
        _bonded_pair(lattice=5.5),
        host_structure(species=("Si", "Si", "Si"), positions=((0, 0, 0), (0.4, 0, 0), (0, 0.4, 0))),
    ],
    ids=["species", "cell", "atom-count"],
)
def test_a_topology_change_rebuilds_the_scene(bonded_viewer, monkeypatch, changed) -> None:
    viewer = bonded_viewer
    renders = count_scene_renders(monkeypatch)

    viewer.set_structure_preserving_view(changed)

    assert renders[viewer] == 1
    assert len(viewer._atoms) == len(changed.species)
    assert list(viewer._atoms.get_chemical_symbols()) == list(changed.species)


def test_preserving_the_view_still_clears_overlays_when_atoms_could_move(
    bonded_viewer, monkeypatch
) -> None:
    viewer = bonded_viewer
    viewer.set_point_overlays(_point_overlay())
    renders = count_scene_renders(monkeypatch)

    viewer.set_structure_preserving_view(
        _bonded_pair(positions=((0.1, 0.2, 0.3), (0.45, 0.2, 0.3)))
    )

    assert renders[viewer] == 1
    assert viewer._pyvista_point_overlay_actors == []


def test_a_move_while_hidden_draws_once_when_shown(bonded_viewer, monkeypatch) -> None:
    viewer = bonded_viewer
    viewer.hide()
    counter = _RenderCounter(viewer, monkeypatch)
    moved = _bonded_pair(positions=((0.1, 0.2, 0.3), (0.45, 0.2, 0.3)))

    viewer.set_structure_preserving_view(moved)
    assert counter.main == 0

    viewer.show()
    assert counter.main == 1
    assert counter.add_mesh_render_flags == []
    assert np.allclose(viewer._pyvista_pick_actor.mesh.points[1], [2.25, 1.0, 1.5])


def test_swapping_two_bonded_atoms_turns_each_half_bond_around(bonded_viewer, monkeypatch) -> None:
    viewer = bonded_viewer
    counter = _RenderCounter(viewer, monkeypatch)
    swapped = _bonded_pair(positions=((0.5, 0.2, 0.3), (0.1, 0.2, 0.3)))

    viewer.set_structure_preserving_view(swapped)

    assert counter.add_mesh_render_flags == []
    expected = np.asarray(swapped.fractional_positions) * 5.0
    midpoint = expected.mean(axis=0)
    first_half, second_half = viewer._pyvista_bond_actors
    _assert_cylinder_between(first_half, expected[0], midpoint)
    _assert_cylinder_between(second_half, midpoint, expected[1])


def test_two_atoms_moved_onto_one_point_draw_no_bond(bonded_viewer) -> None:
    viewer = bonded_viewer

    viewer.set_structure_preserving_view(_bonded_pair(positions=((0.3, 0.2, 0.3), (0.3, 0.2, 0.3))))

    assert viewer._pyvista_bond_actors == []
    assert viewer._pyvista_bond_pieces == {}


def test_atomic_structure_viewer_named_views_follow_lattice_vectors_for_hexagonal_cell(
    qapp,
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(hexagonal_magnesium_oxide_host())

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


def test_atomic_structure_viewer_controls_default_states_and_reset_button_width(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

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
        canvas.ensure_plotter()
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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    # Lazy: the timer is not started in __init__, so an unshown viewer never polls.
    assert not viewer._pyvista_camera_timer.isActive()
    viewer.showEvent(QShowEvent())
    assert viewer._pyvista_camera_timer.isActive()
    viewer.hideEvent(QHideEvent())
    assert not viewer._pyvista_camera_timer.isActive()
    viewer.showEvent(QShowEvent())
    assert viewer._pyvista_camera_timer.isActive()

    widget.close()


def test_atomic_structure_viewer_resyncs_camera_on_show(qapp, monkeypatch) -> None:
    """showEvent must poll the camera once when it restarts the timer.

    The poll timer is the only thing that syncs the axes/triad to the camera.
    Because it is paused while hidden, a camera/structure change made off-screen
    would otherwise leave a stale gizmo until the first tick; a one-shot poll on
    show avoids that flicker.
    """

    from PySide6.QtGui import QHideEvent, QShowEvent

    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer
    viewer.showEvent(QShowEvent())
    viewer.hideEvent(QHideEvent())

    polls = 0
    original = viewer._poll_pyvista_camera

    def _counting() -> None:
        nonlocal polls
        polls += 1
        original()

    monkeypatch.setattr(viewer, "_poll_pyvista_camera", _counting)
    viewer.showEvent(QShowEvent())
    assert viewer._pyvista_camera_timer.isActive()
    assert polls == 1

    # An already-visible viewer should not re-poll on a redundant show.
    viewer.showEvent(QShowEvent())
    assert polls == 1

    widget.close()


def test_atomic_structure_viewer_requires_pyvista_modules(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mufinder.app.atomic_structure_viewer._pyvista_installed", lambda: False)

    with pytest.raises(RuntimeError, match="requires PyVista"):
        AtomicStructureViewer()


def test_pyvista_installed_checks_without_importing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    import mufinder.app.atomic_structure_viewer as module

    assert module._pyvista_installed()
    found = {"pyvista": object(), "pyvistaqt": None}
    monkeypatch.setattr(importlib.util, "find_spec", found.get)
    assert not module._pyvista_installed()


def test_atomic_structure_viewer_selection_label_uses_fixed_single_line_height(qapp) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    viewer = widget._viewer
    assert viewer._selection_label.wordWrap() is False
    assert viewer._selection_label.minimumHeight() == viewer._selection_label.maximumHeight()

    viewer.select_atom(1)

    assert "#2 O" in viewer._selection_label.text()
    assert viewer._selection_label.toolTip() == viewer._selection_label.text()
    assert viewer._selection_label.minimumHeight() == viewer._selection_label.maximumHeight()
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


def test_atomic_structure_viewer_blank_click_clears_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())

    viewer = widget._viewer
    viewer.select_atom(1)
    assert "#2 O" in viewer._selection_label.text()

    install_fake_picker(viewer, monkeypatch, picked_point=None, position=(9, 11))
    viewer._on_pyvista_left_button_press()
    assert "#2 O" in viewer._selection_label.text()
    viewer._on_pyvista_left_button_release()

    assert viewer._selection_label.text() == "Selected atom: -"
    widget.close()


def test_atomic_structure_viewer_shift_selection_adds_and_removes_atoms(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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


def test_atomic_structure_viewer_shift_selection_uses_latched_modifier_when_pick_event_loses_it(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    viewer.select_atom(0)
    monkeypatch.setattr(viewer, "_selection_modifier_active", lambda _event=None: False)
    viewer._latched_selection_modifier = True

    viewer._apply_pick_selection(1)

    assert viewer._selected_indexes == [0, 1]
    assert "Selected atoms: 2" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_shift_selection_uses_native_mouse_modifier_when_backend_event_loses_it(
    qapp,
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    viewer.select_atoms([0, 1])
    assert viewer._selected_indexes == [0, 1]

    install_fake_picker(viewer, monkeypatch, picked_point=None, position=(9, 11))
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


def test_atomic_structure_viewer_pyvista_pick_selects_nearest_atom(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.set_structure(magnetic_silicon_oxide_host())
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
    widget.set_structure(magnetic_silicon_oxide_host())
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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    viewer.select_atom(0)
    picker, _position = install_fake_picker(
        viewer, monkeypatch, picked_point=(1.1, 1.0, 1.0), shift=True
    )
    viewer._native_selection_modifier = False
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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    viewer.select_atom(1)
    picker, _position = install_fake_picker(
        viewer, monkeypatch, picked_point=None, position=(9, 11)
    )
    viewer._native_selection_modifier = False

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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    viewer.select_atom(0)
    picker, current_position = install_fake_picker(
        viewer, monkeypatch, picked_point=(1.1, 1.0, 1.0), position=(5, 7)
    )
    viewer._pyvista_atom_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    viewer._pyvista_atom_radii = np.array([0.6, 0.45], dtype=float)
    viewer._pyvista_atom_lookup_indices = np.array([0, 1], dtype=int)

    viewer._on_pyvista_left_button_press()
    current_position["value"] = (22, 29)
    viewer._on_pyvista_left_button_release()

    assert picker.calls == []
    assert viewer._selected_indexes == [0]
    assert "#1 Si" in viewer._selection_label.text()
    widget.close()


def test_atomic_structure_viewer_pyvista_camera_poll_redraws_triad_canvas(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())

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
    monkeypatch.setattr(
        viewer,
        "_draw_crystallographic_axes",
        lambda: calls.__setitem__("triad", calls["triad"] + 1),
    )

    viewer._poll_pyvista_camera()

    assert calls["triad"] == 1
    widget.close()


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


def test_atomic_structure_viewer_reuses_pyvista_camera_snapshot_on_redraw(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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


def test_atomic_structure_viewer_set_structure_preserving_view_keeps_camera_limits_and_spin_toggle(
    qapp,
) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(magnetic_silicon_oxide_host())
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


def test_atomic_structure_viewer_update_structure_geometry_preserving_view_keeps_camera_and_updates_positions(
    qapp,
) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(magnetic_silicon_oxide_host())
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
    assert np.allclose(
        viewer._atoms.get_scaled_positions(wrap=True),
        np.asarray(updated.fractional_positions, dtype=float),
    )
    viewer.close()


def test_atomic_structure_viewer_pyvista_zoom_only_poll_updates_camera_snapshot(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = HostStructureWidget()
    widget.show()
    widget.set_structure(magnetic_silicon_oxide_host())
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


def test_atomic_structure_viewer_theme_refresh_rerenders_scene(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = AtomicStructureViewer()
    viewer.show()
    calls = {"store": 0, "render": 0, "canvas": 0, "triad": 0}

    draws: list[bool] = []

    class _Canvas:
        # A shown viewer's canvas has its plotter.
        plotter = object()

        def refresh_theme(self, *, draw: bool = True) -> None:
            calls["canvas"] += 1
            draws.append(draw)

    class _Triad:
        def refresh_theme(self, *, draw: bool = True) -> None:
            calls["triad"] += 1
            draws.append(draw)

    viewer._canvas = _Canvas()
    viewer._triad_canvas = _Triad()
    monkeypatch.setattr(
        viewer, "_store_view", lambda: calls.__setitem__("store", calls["store"] + 1)
    )
    monkeypatch.setattr(viewer, "_render", lambda: calls.__setitem__("render", calls["render"] + 1))

    viewer.refresh_theme()

    assert calls == {"store": 1, "render": 1, "canvas": 1, "triad": 1}
    # The scene rebuild draws the new palette; the canvases do not draw first.
    assert draws == [False, False]
    viewer.close()


def test_atomic_structure_viewer_coalesces_theme_change_events(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = AtomicStructureViewer()
    calls = {"refresh": 0}
    monkeypatch.setattr(
        viewer, "refresh_theme", lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
    )

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
    widget.set_structure(magnetic_silicon_oxide_host())
    viewer = widget._viewer

    class _FakePlotter:
        def __init__(self) -> None:
            self.render_timer = FakeRenderTimer()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    main_plotter = _FakePlotter()
    triad_plotter = _FakePlotter()
    viewer._backend_name = "pyvista"
    viewer._pyvista_camera_timer = FakeRenderTimer()
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
