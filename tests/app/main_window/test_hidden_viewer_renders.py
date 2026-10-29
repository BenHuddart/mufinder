"""Structure viewers on pages that are not showing do not render.

A viewer that is not visible records each change and draws it once when it is
shown, so opening a project or switching the active host renders only the
viewer on screen. A viewer creates its VTK render windows on first show, so
building the window creates none. These tests count full scene rebuilds per viewer through the
window paths a user drives: opening a saved project, choosing another host and
navigating to a page.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.main_window import MainWindow
from mufinder.services.project_service import ProjectService
from tests.support.main_window import open_project_with_host, select_route
from tests.support.pyvista_fakes import count_created_interactors, count_scene_renders
from tests.support.structures import host_structure


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count scene rebuilds per viewer, keyed by the viewer."""

    return count_scene_renders(monkeypatch)


@pytest.fixture
def window(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def _saved_project_with_hosts(tmp_path: Path, count: int) -> tuple[Path, list]:
    service = ProjectService()
    document = service.create_project(tmp_path / "hosts.mufinder", "Hosts")
    hosts = [host_structure(label=f"Host {index}", lattice=4.0 + index) for index in range(count)]
    for host in hosts:
        service.add_structure(document, host)
    document.project.host_structure_id = hosts[0].id
    service.save_project(document)
    return document.project_root, hosts


def test_building_the_window_renders_no_viewer(qapp, renders: Counter) -> None:
    window = MainWindow()
    try:
        window.show()
        qapp.processEvents()

        assert window.findChildren(AtomicStructureViewer)
        assert sum(renders.values()) == 0
    finally:
        window.close()


def test_building_the_window_creates_no_render_window_until_a_viewer_page_shows(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, renders: Counter
) -> None:
    interactors = count_created_interactors(monkeypatch)
    window = MainWindow()
    try:
        window.show()
        qapp.processEvents()
        assert window.findChildren(AtomicStructureViewer)
        assert interactors == []

        open_project_with_host(window, tmp_path)
        qapp.processEvents()
        assert interactors == []

        select_route(window, "host_structure")
        qapp.processEvents()

        host_viewer = window._host_structure_view._viewer
        # The host viewer's scene and triad, rendered once; no other viewer.
        assert len(interactors) == 2
        assert {host_viewer._canvas.plotter, host_viewer._triad_canvas.plotter} == set(interactors)
        assert renders == Counter({host_viewer: 1})
    finally:
        window.close()


def test_opening_a_project_on_a_page_without_a_viewer_renders_no_viewer(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter
) -> None:
    root, _hosts = _saved_project_with_hosts(tmp_path, 1)

    assert window.open_project(root)
    qapp.processEvents()

    page = window._main_stack.currentWidget()
    assert not page.findChildren(AtomicStructureViewer)
    assert sum(renders.values()) == 0


def test_switching_the_host_renders_only_the_visible_viewer_once(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter
) -> None:
    root, hosts = _saved_project_with_hosts(tmp_path, 2)
    assert window.open_project(root)
    select_route(window, "host_structure")
    qapp.processEvents()
    host_viewer = window._host_structure_view._viewer
    generation_viewer = window._muon_generation_view._viewer
    assert host_viewer.isVisible()
    assert not generation_viewer.isVisible()
    renders.clear()

    window._host_structure_view.active_host_change_requested.emit(str(hosts[1].id))
    qapp.processEvents()

    assert renders == Counter({host_viewer: 1})
    assert len(host_viewer._atoms) == len(hosts[1].species)

    select_route(window, "muon_candidates")
    qapp.processEvents()

    assert renders[generation_viewer] == 1
    assert renders[host_viewer] == 1
