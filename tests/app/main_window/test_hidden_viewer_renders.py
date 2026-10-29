"""Structure viewers on pages that are not showing do not render.

A viewer that is not visible records each change and draws it once when it is
shown, so opening a project or switching the active host renders only the
viewer on screen. These tests count full scene rebuilds per viewer through the
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
from tests.support.main_window import select_route
from tests.support.structures import host_structure


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count scene rebuilds per viewer, keyed by the viewer."""

    counts: Counter = Counter()
    original = AtomicStructureViewer._render_pyvista

    def counted(self) -> None:
        counts[self] += 1
        original(self)

    monkeypatch.setattr(AtomicStructureViewer, "_render_pyvista", counted)
    return counts


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
