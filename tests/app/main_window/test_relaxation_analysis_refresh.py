"""The Relaxation Analysis page is rebuilt only when it is showing.

Each analysis workspace tab holds a 3D structure viewer, so rebuilding the page
while another page is showing froze the window on large projects. These tests
count the page's rebuilds through the window's public paths: opening a project,
navigating to the page, and a job-status refresh.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import (
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.services.project_service import ProjectService
from tests.support.structures import host_structure


WORKSPACE_NAMES = ("First analysis", "Second analysis", "Third analysis")


@pytest.fixture
def rebuilds(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count ``set_context`` and ``_rebuild_tabs`` calls on every analysis page."""

    counts = {"set_context": 0, "_rebuild_tabs": 0}
    for name in counts:
        original = getattr(RelaxationAnalysisWidget, name)

        def counted(self, *args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(RelaxationAnalysisWidget, name, counted)
    return counts


def _reset(counts: dict[str, int]) -> None:
    for name in counts:
        counts[name] = 0


def _saved_project_with_workspaces(tmp_path: Path) -> tuple[Path, list[dict]]:
    service = ProjectService()
    document = service.create_project(tmp_path / "analysis.mufinder", "Analysis Doc")
    host = host_structure()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    workspaces = [
        default_workspace_state(
            name=name,
            charge_state=POSITIVE_MUON_CHARGE_STATE,
            host_structure_id=host.id,
            job_ids=[uuid4()],
        )
        for name in WORKSPACE_NAMES
    ]
    document.ui_state["relaxation_analysis_workspaces"] = workspaces
    service.save_project(document)
    return document.project_root, workspaces


def test_opening_a_project_does_not_build_the_hidden_analysis_page(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, _workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()

    assert window.open_project(root)

    assert window._main_stack.currentWidget() is not window._relaxation_analysis_view
    assert rebuilds == {"set_context": 0, "_rebuild_tabs": 0}


def test_showing_the_analysis_page_builds_it_once(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()
    assert window.open_project(root)

    window._set_active_route("muon_analysis")

    view = window._relaxation_analysis_view
    assert window._main_stack.currentWidget() is view
    assert rebuilds == {"set_context": 1, "_rebuild_tabs": 1}
    assert view._workspace_tabs.count() == len(workspaces)

    # Leaving and coming back with nothing changed does not rebuild it again.
    window._set_active_route("host_structure")
    window._set_active_route("muon_analysis")
    assert rebuilds == {"set_context": 1, "_rebuild_tabs": 1}


def test_a_workspace_route_builds_the_page_before_focusing_its_tab(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()
    assert window.open_project(root)
    second = str(workspaces[1]["id"])

    window._set_active_route(window._relaxation_analysis_workspace_route(second))

    assert window._relaxation_analysis_view.current_workspace_id() == second
    assert rebuilds["set_context"] == 1


def test_a_job_status_refresh_does_not_rebuild_the_hidden_analysis_page(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, _workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()
    assert window.open_project(root)
    window._set_active_route("muon_analysis")
    window._set_active_route("host_structure")
    _reset(rebuilds)

    # The path a local job status change and a dashboard flush both take.
    window._refresh_jobs_table()

    assert rebuilds == {"set_context": 0, "_rebuild_tabs": 0}

    # The skipped refresh is not lost: the page catches up when it is shown.
    window._set_active_route("muon_analysis")
    assert rebuilds == {"set_context": 1, "_rebuild_tabs": 1}


def test_a_job_status_refresh_keeps_the_showing_analysis_page_current(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()
    assert window.open_project(root)
    window._set_active_route("muon_analysis")
    _reset(rebuilds)

    window._refresh_jobs_table()

    assert rebuilds == {"set_context": 1, "_rebuild_tabs": 1}
    assert window._relaxation_analysis_view._workspace_tabs.count() == len(workspaces)


def test_a_project_change_rebuilds_the_analysis_page_at_most_once(
    qapp, tmp_path: Path, rebuilds: dict[str, int]
) -> None:
    root, _workspaces = _saved_project_with_workspaces(tmp_path)
    window = MainWindow()
    assert window.open_project(root)
    window._set_active_route("muon_analysis")
    _reset(rebuilds)

    window._refresh_ui_after_project_change()
    window._set_active_route("muon_analysis")

    assert rebuilds == {"set_context": 1, "_rebuild_tabs": 1}
