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
from mufinder.app.supercell_dialog import SupercellPreviewDialog
from tests.support.structures import host_structure

_magnetic_host = partial(
    host_structure,
    lattice=4.0,
    species=("Fe", "O"),
    properties={"pbc": [True, True, True], "atomic_spins": [1.0, -1.0]},
)


class _RenderCounter:
    """Count ``render`` calls on a viewer's main and triad plotters."""

    def __init__(self, viewer: AtomicStructureViewer, monkeypatch: pytest.MonkeyPatch) -> None:
        self.main = 0
        self.triad = 0
        self.add_mesh_render_flags: list[object] = []
        plotter = viewer._canvas.plotter
        triad_plotter = viewer._triad_canvas.plotter
        original_add_mesh = plotter.add_mesh

        def count_main() -> None:
            self.main += 1

        def count_triad() -> None:
            self.triad += 1

        def recording_add_mesh(mesh, **kwargs):
            self.add_mesh_render_flags.append(kwargs.get("render", True))
            return original_add_mesh(mesh, **kwargs)

        monkeypatch.setattr(plotter, "render", count_main)
        monkeypatch.setattr(triad_plotter, "render", count_triad)
        monkeypatch.setattr(plotter, "add_mesh", recording_add_mesh)

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


def test_supercell_preview_shows_structure_and_spins_with_one_render(qapp, monkeypatch) -> None:
    dialog = SupercellPreviewDialog(
        structure=_magnetic_host(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
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
