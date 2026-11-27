"""Behaviour tests for deleting a structural-relaxation analysis workspace."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from uuid import uuid4

from PySide6.QtWidgets import QPushButton

from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import (
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.models import Structure
from mufinder.services.analysis_workspaces import analysis_workspaces, set_analysis_workspaces
from mufinder.services.muon_sites_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    store_muon_sites_summary_snapshot,
)
from mufinder.services.project_summary import relaxation_analysis_complete
from tests.support.main_window import select_route, window_with_host
from tests.support.structures import host_structure


def _workspace(
    host: Structure, name: str, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE
) -> dict:
    return default_workspace_state(
        name=name,
        charge_state=charge_state,
        host_structure_id=host.id,
        job_ids=[uuid4()],
    )


def _snapshot(workspace: dict, host: Structure) -> dict:
    return {
        "charge_state": workspace["charge_state"],
        "workspace_id": workspace["id"],
        "workspace_name": workspace["name"],
        "host_structure_id": str(host.id),
        "sites": [],
    }


_window_with_host = partial(window_with_host, filename="analysis.mufinder", name="Analysis Doc")


def test_delete_button_emits_workspace_id(qapp) -> None:
    host = host_structure()
    workspace = _workspace(host, "Preliminary analysis")
    widget = RelaxationAnalysisWidget()
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=[],
            structures={},
            host_structure=host,
            comparison_options=[],
        )
        captured: list[str] = []
        widget.delete_requested.connect(captured.append)
        button = widget.findChild(QPushButton, "analysisWorkspaceDeleteButton")
        assert button is not None
        button.click()
        assert captured == [str(workspace["id"])]
    finally:
        widget.deleteLater()


def test_delete_removes_only_the_chosen_workspace_and_persists(
    qapp, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window, service, doc, host = _window_with_host(tmp_path)
    try:
        stale = _workspace(host, "Preliminary analysis")
        kept = _workspace(host, "Full analysis")
        set_analysis_workspaces(window._document, [stale, kept])
        window._refresh_relaxation_analysis_context()
        window._refresh_navigator()
        window._set_active_route("muon_analysis")

        window._analysis_workspaces_controller.delete_workspace(str(stale["id"]))

        assert [w["id"] for w in analysis_workspaces(window._document)] == [kept["id"]]
        assert window._relaxation_analysis_view._workspace_tabs.count() == 1
        prompt = confirmed_destructive_actions[-1]
        assert prompt["text"] == "Delete analysis 'Preliminary analysis'?"
        assert prompt["accept_label"] == "Delete"
        assert "relaxations it uses are kept" in prompt["detail"]
        assert "pin" not in prompt["detail"]
        reloaded = service.load_project(doc.project_root)
        assert [w["id"] for w in reloaded.ui_state["relaxation_analysis_workspaces"]] == [
            kept["id"]
        ]
    finally:
        window.close()


def test_declining_the_confirmation_keeps_the_workspace(qapp, tmp_path: Path, monkeypatch) -> None:
    window, _service, _doc, host = _window_with_host(tmp_path)
    try:
        workspace = _workspace(host, "Preliminary analysis")
        set_analysis_workspaces(window._document, [workspace])
        monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

        window._analysis_workspaces_controller.delete_workspace(str(workspace["id"]))

        assert [w["id"] for w in analysis_workspaces(window._document)] == [workspace["id"]]
    finally:
        window.close()


def test_delete_removes_a_summary_pin_taken_from_the_workspace(
    qapp, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        stale = _workspace(host, "Preliminary analysis")
        other = _workspace(host, "Muonium analysis", charge_state=MUONIUM_CHARGE_STATE)
        set_analysis_workspaces(window._document, [stale, other])
        store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, _snapshot(stale, host))
        store_muon_sites_summary_snapshot(doc, MUONIUM_CHARGE_STATE, _snapshot(other, host))

        window._analysis_workspaces_controller.delete_workspace(str(stale["id"]))

        snapshots = doc.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY]
        assert POSITIVE_MUON_CHARGE_STATE not in snapshots
        assert snapshots[MUONIUM_CHARGE_STATE]["workspace_id"] == other["id"]
        assert "pin is removed" in confirmed_destructive_actions[-1]["detail"]
    finally:
        window.close()


def test_deleting_the_last_workspace_clears_the_analysis_stage(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        workspace = _workspace(host, "Preliminary analysis")
        set_analysis_workspaces(window._document, [workspace])
        window._refresh_navigator()
        window._set_active_route("muon_analysis")
        assert relaxation_analysis_complete(doc)

        window._analysis_workspaces_controller.delete_workspace(str(workspace["id"]))

        assert not relaxation_analysis_complete(doc)
        assert window._relaxation_analysis_view._workspace_tabs.count() == 0
    finally:
        window.close()


# --- Navigator rows and right-click delete ---------------------------------


def _analysis_route(window: MainWindow, workspace: dict) -> str:
    return window._relaxation_analysis_workspace_route(str(workspace["id"]))


def _window_with_workspaces(tmp_path: Path, *names: str) -> tuple[MainWindow, object, list[dict]]:
    window, _service, doc, host = _window_with_host(tmp_path)
    workspaces = [_workspace(host, name) for name in names]
    set_analysis_workspaces(window._document, workspaces)
    window._refresh_relaxation_analysis_context()
    window._refresh_navigator()
    return window, doc, workspaces


def test_each_workspace_has_a_navigator_row_that_opens_its_tab(qapp, tmp_path: Path) -> None:
    window, _doc, (first, second) = _window_with_workspaces(
        tmp_path, "Preliminary analysis", "Full analysis"
    )
    try:
        labels = [(row.indent, row.label) for row in window._project_navigator.rows()]
        analysis = labels.index((1, "Analysis"))
        assert labels[analysis + 1 : analysis + 3] == [
            (2, "Preliminary analysis"),
            (2, "Full analysis"),
        ]

        select_route(window, _analysis_route(window, second))

        assert window._main_stack.currentWidget() is window._relaxation_analysis_view
        assert window._relaxation_analysis_view.current_workspace_id() == second["id"]
    finally:
        window.close()


def test_switching_analysis_tabs_selects_the_matching_row(qapp, tmp_path: Path) -> None:
    window, _doc, (first, second) = _window_with_workspaces(
        tmp_path, "Preliminary analysis", "Full analysis"
    )
    try:
        window._set_active_route(window._relaxation_analysis_workspace_route(str(first["id"])))

        window._relaxation_analysis_view._workspace_tabs.setCurrentIndex(1)

        assert (
            window._project_navigator.active_route()
            == window._relaxation_analysis_workspace_route(str(second["id"]))
        )
    finally:
        window.close()


def test_renaming_a_workspace_relabels_its_row(qapp, tmp_path: Path) -> None:
    window, _doc, (workspace,) = _window_with_workspaces(tmp_path, "Preliminary analysis")
    try:
        window._set_active_route(window._relaxation_analysis_workspace_route(str(workspace["id"])))
        window._relaxation_analysis_view._rename_workspace(str(workspace["id"]), "Early look")

        assert window._project_navigator.row(_analysis_route(window, workspace)).label == (
            "Early look"
        )
    finally:
        window.close()


def test_navigator_context_menu_deletes_an_analysis(
    qapp, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window, _doc, (stale, kept) = _window_with_workspaces(
        tmp_path, "Preliminary analysis", "Full analysis"
    )
    try:
        # Deleting the node that is showing must land on the Analysis page,
        # not drop back to the project home page.
        window._set_active_route(window._relaxation_analysis_workspace_route(str(stale["id"])))

        menu = window._project_navigator.context_menu(_analysis_route(window, stale))
        assert menu is not None
        assert [action.text() for action in menu.actions()] == ["Delete Analysis…"]
        menu.actions()[0].trigger()

        assert [w["id"] for w in analysis_workspaces(window._document)] == [kept["id"]]
        assert (
            confirmed_destructive_actions[-1]["text"] == "Delete analysis 'Preliminary analysis'?"
        )
        assert not window._project_navigator.has_route(_analysis_route(window, stale))
        assert window._main_stack.currentWidget() is window._relaxation_analysis_view
        assert window._project_navigator.active_route() == "muon_analysis"
    finally:
        window.close()


def test_navigator_context_menu_offers_magnetic_workspace_delete(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    window, doc, _workspaces = _window_with_workspaces(tmp_path)
    try:
        doc.ui_state[window._MAGNETIC_FIELD_WORKSPACES_KEY] = [{"id": "mag-1", "title": "Helix"}]
        window._refresh_navigator()
        deleted: list[str] = []
        monkeypatch.setattr(window, "_delete_magnetic_field_workspace", deleted.append)

        menu = window._project_navigator.context_menu(
            window._magnetic_field_workspace_route("mag-1")
        )
        assert menu is not None
        assert [action.text() for action in menu.actions()] == ["Delete Workspace…"]
        menu.actions()[0].trigger()

        assert deleted == ["mag-1"]
    finally:
        window.close()


def test_workflow_stage_rows_have_no_context_menu(qapp, tmp_path: Path) -> None:
    window, _doc, _workspaces = _window_with_workspaces(tmp_path)
    try:
        for route in ("host_structure", "muon_analysis", "muon_sites"):
            assert window._project_navigator.context_menu(route) is None
    finally:
        window.close()


def test_deleting_one_magnetic_workspace_keeps_another_shown(qapp, tmp_path: Path) -> None:
    window, doc, _workspaces = _window_with_workspaces(tmp_path)
    try:
        doc.ui_state[window._MAGNETIC_FIELD_WORKSPACES_KEY] = [
            {"id": "mag-1", "title": "Helix"},
            {"id": "mag-2", "title": "Cone"},
        ]
        window._sync_magnetic_field_workspace_views()
        window._refresh_navigator()
        shown = window._magnetic_field_workspace_route("mag-2")
        select_route(window, shown)

        window._delete_magnetic_field_workspace("mag-1")

        # The surviving workspace's page is rebuilt; its row and its page stay in step.
        assert window._project_navigator.current_route() == shown
        assert (
            window._main_stack.currentWidget() is window._magnetic_field_workspace_widgets["mag-2"]
        )
    finally:
        window.close()
