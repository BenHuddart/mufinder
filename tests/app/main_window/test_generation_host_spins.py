"""The Generation viewer draws the host as it is now, after a spin or colour edit.

The page reloads its viewer only when what the viewer draws from the host has
changed. A spin edit on Host Structure reaches it at once; while the viewer
shows, the arrows are redrawn in place, keeping the camera, the markers and the
user's Spins choice. Any other property change reloads it with the view kept.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mufinder.app.main_window import MainWindow
from tests.support.main_window import flush_dashboard, open_project_with_host
from tests.support.pyvista_fakes import count_scene_renders
from tests.support.structures import host_structure


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> Counter:
    return count_scene_renders(monkeypatch)


@pytest.fixture
def window(qapp, tmp_path: Path):
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    open_project_with_host(
        window,
        tmp_path,
        host=host_structure(species=("Fe", "Fe"), properties={"atomic_spins": [1.0, -1.0]}),
    )
    qapp.processEvents()
    yield window
    window.close()


def _settle(qapp) -> None:
    for _ in range(3):
        qapp.processEvents()


def _host(window: MainWindow):
    return window._document.structures[window._document.project.host_structure_id]


def _show_generation_with_a_marker(qapp, window: MainWindow):
    """Show Generation with the manual method's preview marker; return its viewer."""

    assert window.show_page("muon_candidates")
    view = window._muon_generation_view
    view._method_combo.setCurrentIndex(view._method_combo.findData("manual"))
    _settle(qapp)
    viewer = view._viewer
    assert viewer._point_overlay_specs
    return viewer


def test_a_host_spin_edit_reaches_the_generation_viewer_without_a_job_event(qapp, window) -> None:
    viewer = _show_generation_with_a_marker(qapp, window)
    assert window.show_page("host_structure")
    _settle(qapp)

    window._host_structure_view._atom_table.item(0, 4).setText("2.0")
    _settle(qapp)
    assert window.show_page("muon_candidates")
    _settle(qapp)

    assert list(viewer._spins) == [2.0, -1.0]
    assert viewer._point_overlay_specs


@pytest.mark.parametrize("arrows_shown", [True, False], ids=["arrows-shown", "arrows-hidden"])
def test_a_spin_edit_redraws_a_showing_viewer_in_place(
    qapp, window: MainWindow, renders: Counter, arrows_shown: bool
) -> None:
    viewer = _show_generation_with_a_marker(qapp, window)
    viewer._view_x_button.click()
    camera = viewer._camera_state
    marker = viewer._point_overlay_specs[0].positions.copy()
    # The host's spins ticked Spins; the user may have hidden the arrows since.
    assert viewer._show_spins.isChecked()
    viewer._show_spins.setChecked(arrows_shown)
    _settle(qapp)
    atoms = viewer._pyvista_atom_actor
    renders.clear()

    _host(window).properties["atomic_spins"][0] = 3.0
    flush_dashboard(window)
    _settle(qapp)

    assert list(viewer._spins) == [3.0, -1.0]
    assert viewer._show_spins.isChecked() is arrows_shown
    if arrows_shown:
        assert list(viewer._pyvista_drawn_spins) == [3.0, -1.0]
    assert renders[viewer] == 0
    assert viewer._pyvista_atom_actor is atoms
    assert viewer._camera_state == camera
    assert np.allclose(viewer._point_overlay_specs[0].positions, marker)

    # A second refresh with nothing changed draws nothing.
    flush_dashboard(window)
    _settle(qapp)
    assert renders[viewer] == 0


def test_spins_on_a_host_without_any_tick_spins_on_generation(
    qapp, window: MainWindow, tmp_path: Path
) -> None:
    _service, _document, host = open_project_with_host(
        window,
        tmp_path / "unpolarised",
        host=host_structure(species=("Fe", "Fe"), properties={"atomic_spins": [0.0, 0.0]}),
    )
    viewer = _show_generation_with_a_marker(qapp, window)
    assert not viewer._show_spins.isChecked()

    host.properties["atomic_spins"][1] = 2.0
    flush_dashboard(window)
    _settle(qapp)

    assert list(viewer._spins) == [0.0, 2.0]
    assert viewer._show_spins.isChecked()


def test_a_host_colour_override_redraws_the_generation_viewer_keeping_the_view(
    qapp, window: MainWindow, renders: Counter
) -> None:
    viewer = _show_generation_with_a_marker(qapp, window)
    viewer._view_x_button.click()
    camera = viewer._camera_state
    renders.clear()

    _host(window).properties["atom_colors"] = ["#ff0000", None]
    flush_dashboard(window)
    _settle(qapp)

    assert renders[viewer] <= 1
    assert np.allclose(viewer._pyvista_all_atom_colors[0], (1.0, 0.0, 0.0))
    assert viewer._camera_state == camera
    assert viewer._point_overlay_specs
