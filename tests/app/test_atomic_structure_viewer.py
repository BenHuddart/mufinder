"""The structure viewer draws each caller change with one render.

Every ``add_mesh`` in a render pass passes ``render=False`` (a real PyVista
plotter otherwise re-renders the scene once per mesh), the cell outline is one
actor, and ``batch_update()`` folds several viewer changes into one render.
The plotter here is the test fake from ``tests/conftest.py``; a counting stub
on its ``render`` stands in for VTK's ``Render``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.services.project_service import ProjectService
from tests.support.pyvista_fakes import (
    FakePlotter,
    count_created_interactors,
    count_scene_renders,
)
from tests.support.structures import host_structure

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
