"""Behaviour tests for deleting a structural-relaxation analysis workspace."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PySide6.QtWidgets import QPushButton

from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import RelaxationAnalysisWidget, default_workspace_state
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService


def _host_structure() -> Structure:
    return Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _workspace(host: Structure, name: str, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE) -> dict:
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


def _window_with_host(tmp_path: Path) -> tuple[MainWindow, ProjectService, object, Structure]:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "analysis.mufinder", "Analysis Doc")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc
    window._refresh_ui_after_project_change()
    return window, service, doc, host


def test_delete_button_emits_workspace_id(qapp) -> None:
    host = _host_structure()
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
        window._set_relaxation_analysis_workspaces([stale, kept])
        window._refresh_relaxation_analysis_context()

        window._delete_relaxation_analysis_workspace(str(stale["id"]))

        assert [w["id"] for w in window._relaxation_analysis_workspaces()] == [kept["id"]]
        assert window._relaxation_analysis_view._workspace_tabs.count() == 1
        prompt = confirmed_destructive_actions[-1]
        assert prompt["text"] == "Delete analysis 'Preliminary analysis'?"
        assert prompt["accept_label"] == "Delete"
        assert "relaxations it uses are kept" in prompt["detail"]
        assert "pin" not in prompt["detail"]
        reloaded = service.load_project(doc.project_root)
        assert [w["id"] for w in reloaded.ui_state["relaxation_analysis_workspaces"]] == [kept["id"]]
    finally:
        window.close()


def test_declining_the_confirmation_keeps_the_workspace(qapp, tmp_path: Path, monkeypatch) -> None:
    window, _service, _doc, host = _window_with_host(tmp_path)
    try:
        workspace = _workspace(host, "Preliminary analysis")
        window._set_relaxation_analysis_workspaces([workspace])
        monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

        window._delete_relaxation_analysis_workspace(str(workspace["id"]))

        assert [w["id"] for w in window._relaxation_analysis_workspaces()] == [workspace["id"]]
    finally:
        window.close()


def test_delete_removes_a_summary_pin_taken_from_the_workspace(
    qapp, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        stale = _workspace(host, "Preliminary analysis")
        other = _workspace(host, "Muonium analysis", charge_state=MUONIUM_CHARGE_STATE)
        window._set_relaxation_analysis_workspaces([stale, other])
        window._store_muon_sites_summary_snapshot(POSITIVE_MUON_CHARGE_STATE, _snapshot(stale, host))
        window._store_muon_sites_summary_snapshot(MUONIUM_CHARGE_STATE, _snapshot(other, host))

        window._delete_relaxation_analysis_workspace(str(stale["id"]))

        snapshots = doc.ui_state[window._MUON_SITES_SUMMARY_SNAPSHOTS_KEY]
        assert POSITIVE_MUON_CHARGE_STATE not in snapshots
        assert snapshots[MUONIUM_CHARGE_STATE]["workspace_id"] == other["id"]
        assert "pin is removed" in confirmed_destructive_actions[-1]["detail"]
    finally:
        window.close()


def test_deleting_the_last_workspace_clears_the_analysis_stage(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        workspace = _workspace(host, "Preliminary analysis")
        window._set_relaxation_analysis_workspaces([workspace])
        assert window._workflow_relaxation_analysis_complete(doc)

        window._delete_relaxation_analysis_workspace(str(workspace["id"]))

        assert not window._workflow_relaxation_analysis_complete(doc)
        assert window._relaxation_analysis_view._workspace_tabs.count() == 0
    finally:
        window.close()


# --- Project tree nodes and right-click delete ----------------------------


def _analysis_item(window: MainWindow, workspace: dict):
    return window._find_tree_item_by_route(window._relaxation_analysis_workspace_route(str(workspace["id"])))


def _window_with_workspaces(tmp_path: Path, *names: str) -> tuple[MainWindow, object, list[dict]]:
    window, _service, doc, host = _window_with_host(tmp_path)
    workspaces = [_workspace(host, name) for name in names]
    window._set_relaxation_analysis_workspaces(workspaces)
    window._refresh_relaxation_analysis_context()
    window._populate_project_tree()
    return window, doc, workspaces


def test_each_workspace_has_a_project_tree_node_that_opens_its_tab(qapp, tmp_path: Path) -> None:
    window, _doc, (first, second) = _window_with_workspaces(tmp_path, "Preliminary analysis", "Full analysis")
    try:
        parent = window._find_tree_item_by_route("muon_analysis")
        assert [parent.child(i).text(0) for i in range(parent.childCount())] == [
            "Preliminary analysis",
            "Full analysis",
        ]
        assert parent.isExpanded()

        window._project_tree.setCurrentItem(_analysis_item(window, second))

        assert window._main_stack.currentWidget() is window._relaxation_analysis_view
        assert window._relaxation_analysis_view.current_workspace_id() == second["id"]
    finally:
        window.close()


def test_switching_analysis_tabs_selects_the_matching_tree_node(qapp, tmp_path: Path) -> None:
    window, _doc, (first, second) = _window_with_workspaces(tmp_path, "Preliminary analysis", "Full analysis")
    try:
        window._set_active_route(window._relaxation_analysis_workspace_route(str(first["id"])))

        window._relaxation_analysis_view._workspace_tabs.setCurrentIndex(1)

        assert window._active_route() == window._relaxation_analysis_workspace_route(str(second["id"]))
    finally:
        window.close()


def test_renaming_a_workspace_relabels_its_tree_node(qapp, tmp_path: Path) -> None:
    window, _doc, (workspace,) = _window_with_workspaces(tmp_path, "Preliminary analysis")
    try:
        window._relaxation_analysis_view._rename_workspace(str(workspace["id"]), "Early look")

        assert _analysis_item(window, workspace).text(0) == "Early look"
    finally:
        window.close()


def test_tree_context_menu_deletes_an_analysis(qapp, tmp_path: Path, confirmed_destructive_actions) -> None:
    window, _doc, (stale, kept) = _window_with_workspaces(tmp_path, "Preliminary analysis", "Full analysis")
    try:
        # Deleting the node that is showing must land on the Analysis page,
        # not drop back to the project home page.
        window._set_active_route(window._relaxation_analysis_workspace_route(str(stale["id"])))

        menu = window._project_tree_context_menu(_analysis_item(window, stale))
        assert menu is not None
        assert [action.text() for action in menu.actions()] == ["Delete Analysis..."]
        menu.actions()[0].trigger()

        assert [w["id"] for w in window._relaxation_analysis_workspaces()] == [kept["id"]]
        assert confirmed_destructive_actions[-1]["text"] == "Delete analysis 'Preliminary analysis'?"
        assert _analysis_item(window, stale) is None
        assert window._main_stack.currentWidget() is window._relaxation_analysis_view
        assert window._active_route() == "muon_analysis"
    finally:
        window.close()


def test_tree_context_menu_offers_magnetic_workspace_delete(qapp, tmp_path: Path, monkeypatch) -> None:
    window, doc, _workspaces = _window_with_workspaces(tmp_path)
    try:
        doc.ui_state[window._MAGNETIC_FIELD_WORKSPACES_KEY] = [{"id": "mag-1", "title": "Helix"}]
        window._populate_project_tree()
        deleted: list[str] = []
        monkeypatch.setattr(window, "_delete_magnetic_field_workspace", deleted.append)

        item = window._find_tree_item_by_route(window._magnetic_field_workspace_route("mag-1"))
        menu = window._project_tree_context_menu(item)
        assert menu is not None
        assert [action.text() for action in menu.actions()] == ["Delete Workspace..."]
        menu.actions()[0].trigger()

        assert deleted == ["mag-1"]
    finally:
        window.close()


def test_workflow_stage_nodes_have_no_context_menu(qapp, tmp_path: Path) -> None:
    window, _doc, _workspaces = _window_with_workspaces(tmp_path)
    try:
        for route in ("host_structure", "muon_analysis", "muon_sites"):
            assert window._project_tree_context_menu(window._find_tree_item_by_route(route)) is None
    finally:
        window.close()
