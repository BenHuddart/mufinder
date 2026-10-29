"""Behaviour tests for the Muon Sites summary page and its navigation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from uuid import uuid4

from PySide6.QtWidgets import QPushButton

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.muon_sites_summary_widget import MuonSitesSummaryWidget
from mufinder.app.relaxation_analysis_widget import (
    CompletedRelaxationAnalysisSource,
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.models import Structure
from mufinder.services.muon_sites_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    muon_sites_summary_snapshots,
    store_muon_sites_summary_snapshot,
    validated_muon_sites_summary_snapshot,
)
from tests.support.main_window import window_with_host
from tests.support.structures import host_structure


def _snapshot(
    workspace_id: str, host: Structure, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE
) -> dict:
    return {
        "charge_state": charge_state,
        "workspace_id": workspace_id,
        "workspace_name": "Positive muon analysis",
        "host_structure_id": str(host.id),
        "sites": [
            {
                "cluster_index": 1,
                "fractional_position": [0.25, 0.25, 0.25],
                "relative_energy_ev": 0.0,
                "wyckoff_label": "8c",
            },
            {
                "cluster_index": 2,
                "fractional_position": [0.5, 0.0, 0.0],
                "relative_energy_ev": 0.12,
                "wyckoff_label": "6b",
            },
        ],
    }


def _clustered_workspace(
    host: Structure, job_id, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE
) -> dict:
    workspace = default_workspace_state(
        name="Positive muon analysis",
        charge_state=charge_state,
        host_structure_id=host.id,
        job_ids=[job_id],
    )
    # Two clusters supplied out of energy order so ranking is exercised.
    workspace["clustering"] = {
        "method": "connected",
        "connected_tolerance_ang": 0.25,
        "hierarchical_t": 0.3,
        "kmeans_k": 1,
        "wyckoff_match_tolerance_ang": 0.25,
        "tolerance_scan": [],
        "result": {
            "cluster_labels": [1, 2],
            "clusters": [
                {
                    "cluster_index": 2,
                    "size": 1,
                    "representative_index": 1,
                    "representative_job_id": str(job_id),
                    "representative_fractional_position": [0.5, 0.0, 0.0],
                    "representative_cartesian_position_ang": [2.5, 0.0, 0.0],
                    "minimum_relative_energy_ev": 0.12,
                    "average_relative_energy_ev": 0.12,
                    "standard_deviation_ev": 0.0,
                    "member_indexes": [1],
                },
                {
                    "cluster_index": 1,
                    "size": 1,
                    "representative_index": 0,
                    "representative_job_id": str(job_id),
                    "representative_fractional_position": [0.25, 0.25, 0.25],
                    "representative_cartesian_position_ang": [1.25, 1.25, 1.25],
                    "minimum_relative_energy_ev": 0.0,
                    "average_relative_energy_ev": 0.0,
                    "standard_deviation_ev": 0.0,
                    "member_indexes": [0],
                },
            ],
            "wyckoff_assignments": [
                {
                    "cluster_index": 1,
                    "match": {
                        "matched": True,
                        "label": "8c",
                        "fractional_position": [0.25, 0.25, 0.25],
                    },
                },
                {
                    "cluster_index": 2,
                    "match": {
                        "matched": True,
                        "label": "6b",
                        "fractional_position": [0.5, 0.0, 0.0],
                    },
                },
            ],
        },
    }
    return workspace


# --- Summary widget --------------------------------------------------------


def test_summary_panel_shows_placeholder_when_nothing_pinned(qapp) -> None:
    widget = MuonSitesSummaryWidget()
    try:
        for charge_state in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            panel = widget._panels[charge_state]
            assert not panel._content.isVisibleTo(panel)
            # The action stays enabled even when empty, adapting into a route to
            # the Analysis tab rather than a dead disabled control.
            assert panel._open_button.isEnabled()
            assert panel._open_button.text() == panel._OPEN_EMPTY_TEXT
            assert not panel._unpin_button.isEnabled()
    finally:
        widget.deleteLater()


def test_summary_empty_open_button_routes_to_analysis_root(qapp) -> None:
    widget = MuonSitesSummaryWidget()
    root_requests: list[bool] = []
    workspace_requests: list[str] = []
    widget.open_analysis_root_requested.connect(lambda: root_requests.append(True))
    widget.open_analysis_requested.connect(workspace_requests.append)
    try:
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        panel._open_button.click()
        assert root_requests == [True]
        assert workspace_requests == []
    finally:
        widget.deleteLater()


def test_summary_panel_populates_table_and_preview(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        assert panel._table.rowCount() == 2
        # Ranked lowest energy first.
        assert panel._table.item(0, panel._ENERGY_COLUMN).text() == "0.00000"
        assert panel._table.item(0, panel._WYCKOFF_COLUMN).text() == "8c"
        assert panel._open_button.isEnabled()
        assert panel._unpin_button.isEnabled()
        # Preview appends one H:mu atom per pinned site, mapped back to clusters.
        assert panel._muon_atom_cluster == {2: 1, 3: 2}
        assert widget._panels[MUONIUM_CHARGE_STATE]._table.rowCount() == 0
    finally:
        widget.deleteLater()


def test_summary_repopulates_when_host_edited_in_place(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        snap = _snapshot("ws-1", host)
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE, snapshot=snap, host_structure=host, placeholder_message="x"
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        first_signature = panel._render_signature
        # Same snapshot + a host mutated in place (same id, new lattice) must not
        # hit the dirty-check early return, or the 3D preview would go stale.
        host.lattice_vectors = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE, snapshot=snap, host_structure=host, placeholder_message="x"
        )
        assert panel._render_signature != first_signature
        # An identical repeat call is still skipped by the dirty check.
        skipped_signature = panel._render_signature
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE, snapshot=snap, host_structure=host, placeholder_message="x"
        )
        assert panel._render_signature == skipped_signature
    finally:
        widget.deleteLater()


def test_summary_updates_source_label_on_workspace_rename(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        snap = _snapshot("ws-1", host)
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE, snapshot=snap, host_structure=host, placeholder_message="x"
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        assert "Positive muon analysis" in panel._summary_labels["Source analysis"].text()
        # Rename the source workspace but keep the same clustered sites; the
        # dirty check must still refresh the "Source analysis" summary row.
        renamed = dict(snap)
        renamed["workspace_name"] = "Renamed analysis"
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=renamed,
            host_structure=host,
            placeholder_message="x",
        )
        assert "Renamed analysis" in panel._summary_labels["Source analysis"].text()
    finally:
        widget.deleteLater()


def test_summary_row_click_emits_focus_site(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    captured: list[tuple[str, int]] = []
    widget.focus_site_requested.connect(lambda wid, cluster: captured.append((wid, cluster)))
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        panel._on_table_item_clicked(panel._table.item(1, panel._ENERGY_COLUMN))
        assert captured == [("ws-1", 2)]
    finally:
        widget.deleteLater()


def test_summary_viewer_pick_emits_focus_site_only_for_muons(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    captured: list[tuple[str, int]] = []
    widget.focus_site_requested.connect(lambda wid, cluster: captured.append((wid, cluster)))
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        # Host atom (index 0) is not a muon: no navigation.
        panel._on_viewer_atom_selected(0)
        assert captured == []
        # Muon atom (index 3) maps to cluster 2.
        panel._on_viewer_atom_selected(3)
        assert captured == [("ws-1", 2)]
    finally:
        widget.deleteLater()


def test_summary_open_and_unpin_buttons_emit(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    opened: list[str] = []
    unpinned: list[str] = []
    widget.open_analysis_requested.connect(opened.append)
    widget.unpin_requested.connect(unpinned.append)
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        panel._open_button.click()
        panel._unpin_button.click()
        assert opened == ["ws-1"]
        assert unpinned == [POSITIVE_MUON_CHARGE_STATE]
    finally:
        widget.deleteLater()


def test_summary_export_cif_button_disabled_until_populated(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        assert not panel._export_cif_button.isEnabled()
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        assert panel._export_cif_button.isEnabled()
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=None,
            host_structure=host,
            placeholder_message="cleared",
        )
        assert not panel._export_cif_button.isEnabled()
    finally:
        widget.deleteLater()


def test_summary_export_cif_writes_host_and_muon_sites(qapp, tmp_path: Path, monkeypatch) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]

        output_path = tmp_path / "sites.cif"
        monkeypatch.setattr(
            "mufinder.app.export_dialogs.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(output_path), "CIF Files (*.cif)"),
        )
        monkeypatch.setattr(
            "mufinder.app.export_dialogs.QMessageBox.information",
            lambda *args, **kwargs: None,
        )

        panel._export_cif_button.click()

        assert output_path.exists()
        text = output_path.read_text(encoding="utf-8")
        assert "Si  Si1" in text
        assert "H   Mu1" in text
        assert "H   Mu2" in text
    finally:
        widget.deleteLater()


def test_summary_export_cif_warns_when_site_coincides_with_host_atom(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    host = host_structure()
    snapshot = _snapshot("ws-1", host)
    # Overlay the first pinned site directly onto the host's Si atom at (0,0,0).
    snapshot["sites"][0]["fractional_position"] = [0.0, 0.0, 0.0]
    widget = MuonSitesSummaryWidget()
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=snapshot,
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]

        output_path = tmp_path / "sites.cif"
        monkeypatch.setattr(
            "mufinder.app.export_dialogs.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(output_path), "CIF Files (*.cif)"),
        )
        warned: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.export_dialogs.QMessageBox.warning",
            lambda _parent, title, message: warned.append(f"{title}: {message}"),
        )

        panel._export_cif_button.click()

        assert output_path.exists()
        assert len(warned) == 1
        assert warned[0].startswith("Muon sites coincide with host atoms: ")
        assert "Mu1" in warned[0]
    finally:
        widget.deleteLater()


def test_summary_export_cif_cancelled_dialog_writes_nothing(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        monkeypatch.setattr(
            "mufinder.app.export_dialogs.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: ("", ""),
        )
        panel._export_cif_button.click()
    finally:
        widget.deleteLater()


def test_summary_panel_clears_back_to_placeholder(qapp) -> None:
    host = host_structure()
    widget = MuonSitesSummaryWidget()
    try:
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=_snapshot("ws-1", host),
            host_structure=host,
            placeholder_message="empty",
        )
        widget.set_panel(
            POSITIVE_MUON_CHARGE_STATE,
            snapshot=None,
            host_structure=host,
            placeholder_message="cleared",
        )
        panel = widget._panels[POSITIVE_MUON_CHARGE_STATE]
        assert panel._table.rowCount() == 0
        assert panel._placeholder.text() == "cleared"
        assert panel._open_button.text() == panel._OPEN_EMPTY_TEXT
        assert panel._open_button.isEnabled()
    finally:
        widget.deleteLater()


# --- Analysis widget focus + pin ------------------------------------------


def test_focus_workspace_reveals_cluster_and_reports_missing(qapp) -> None:
    host = host_structure()
    job_a, job_b = uuid4(), uuid4()
    workspace = _clustered_workspace(host, job_a)
    # Two rendered sites (distinct energies keep the energy-sorted order stable)
    # aligned to cluster labels [1, 2] so both cluster headers exist to reveal.
    workspace["job_ids"] = [str(job_a), str(job_b)]
    workspace["clustering"]["result"]["cluster_labels"] = [1, 2]
    sources = [
        CompletedRelaxationAnalysisSource(
            job_id=job_a,
            job_label="Site A",
            charge_state=POSITIVE_MUON_CHARGE_STATE,
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=(0.25, 0.25, 0.25),
            total_energy_ev=-10.0,
        ),
        CompletedRelaxationAnalysisSource(
            job_id=job_b,
            job_label="Site B",
            charge_state=POSITIVE_MUON_CHARGE_STATE,
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=(0.5, 0.0, 0.0),
            total_energy_ev=-9.0,
        ),
    ]
    widget = RelaxationAnalysisWidget()
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=sources,
            structures={host.id: host},
            host_structure=host,
            comparison_options=[],
        )
        workspace_id = str(workspace["id"])
        assert widget.focus_workspace(workspace_id) is True
        assert widget.focus_workspace(workspace_id, cluster_index=1) is True
        # cluster_index <= 0 means "no specific cluster": focus the workspace
        # without reporting a false unresolved target.
        assert widget.focus_workspace(workspace_id, cluster_index=0) is True
        # Unknown cluster / workspace report an unresolved target.
        assert widget.focus_workspace(workspace_id, cluster_index=99) is False
        assert widget.focus_workspace("does-not-exist") is False
    finally:
        widget.deleteLater()


def test_pin_button_emits_pin_summary_requested(qapp) -> None:
    host = host_structure()
    job_id = uuid4()
    workspace = _clustered_workspace(host, job_id)
    source = CompletedRelaxationAnalysisSource(
        job_id=job_id,
        job_label="Site A",
        charge_state=POSITIVE_MUON_CHARGE_STATE,
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.5, 0.0, 0.0),
        total_energy_ev=-10.0,
    )
    widget = RelaxationAnalysisWidget()
    captured: list[str] = []
    widget.pin_summary_requested.connect(captured.append)
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=[source],
            structures={host.id: host},
            host_structure=host,
            comparison_options=[],
        )
        pin_button = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Pin to Summary"
        )
        pin_button.click()
        assert captured == [str(workspace["id"])]
    finally:
        widget.deleteLater()


# --- Main window integration ----------------------------------------------


_window_with_host = partial(window_with_host, filename="summary.mufinder", name="Summary Doc")


def test_pin_builds_ranked_snapshot_and_populates_panel(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        job_id = uuid4()
        workspace = _clustered_workspace(host, job_id)
        window._set_relaxation_analysis_workspaces([workspace])

        window._on_muon_sites_pin_requested(str(workspace["id"]))

        stored = doc.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY]
        assert POSITIVE_MUON_CHARGE_STATE in stored
        sites = stored[POSITIVE_MUON_CHARGE_STATE]["sites"]
        # Ranked by relative energy (lowest first) even though clusters were
        # supplied out of order.
        assert [site["cluster_index"] for site in sites] == [1, 2]
        assert sites[0]["wyckoff_label"] == "8c"
        assert window._main_stack.currentWidget() is window._muon_sites_summary_view
        panel = window._muon_sites_summary_view._panels[POSITIVE_MUON_CHARGE_STATE]
        assert panel._table.rowCount() == 2
    finally:
        window.close()


def test_validated_snapshot_dropped_when_host_changes(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, _snapshot("ws-1", host))
        assert validated_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE) is not None
        # Point the project at a different host: the decoupled snapshot is stale.
        doc.project.host_structure_id = uuid4()
        assert validated_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE) is None
    finally:
        window.close()


def test_unpin_removes_snapshot(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, _snapshot("ws-1", host))
        window._on_unpin_muon_sites_requested(POSITIVE_MUON_CHARGE_STATE)
        assert POSITIVE_MUON_CHARGE_STATE not in doc.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY]
    finally:
        window.close()


def test_pin_without_clustering_is_rejected(qapp, tmp_path: Path, monkeypatch) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        workspace = default_workspace_state(
            name="Positive muon analysis",
            charge_state=POSITIVE_MUON_CHARGE_STATE,
            host_structure_id=host.id,
            job_ids=[uuid4()],
        )
        window._set_relaxation_analysis_workspaces([workspace])
        # Suppress the informational dialog in the headless test.
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: None,
        )
        window._on_muon_sites_pin_requested(str(workspace["id"]))
        assert muon_sites_summary_snapshots(doc) == {}
    finally:
        window.close()


def test_visible_summary_drops_stale_pin_on_host_change(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        workspace = _clustered_workspace(host, uuid4())
        window._set_relaxation_analysis_workspaces([workspace])
        window._on_muon_sites_pin_requested(str(workspace["id"]))
        panel = window._muon_sites_summary_view._panels[POSITIVE_MUON_CHARGE_STATE]
        assert panel._table.rowCount() == 2
        assert window._main_stack.currentWidget() is window._muon_sites_summary_view
        # Switching the active host must drop the now-stale pin from the visible
        # summary (its snapshot is keyed to the previous host id).
        doc.project.host_structure_id = uuid4()
        window._refresh_muon_sites_summary()
        assert panel._table.rowCount() == 0
    finally:
        window.close()


def test_hidden_summary_refresh_fills_the_panel_and_renders_on_show(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        window.show()
        assert window._main_stack.currentWidget() is not window._muon_sites_summary_view
        store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, _snapshot("ws-1", host))
        panel = window._muon_sites_summary_view._panels[POSITIVE_MUON_CHARGE_STATE]
        renders: list[object] = []
        original = AtomicStructureViewer._render_pyvista

        def counted(viewer) -> None:
            renders.append(viewer)
            original(viewer)

        monkeypatch.setattr(AtomicStructureViewer, "_render_pyvista", counted)

        # Behind another page the panel is filled, but its viewer waits to be shown.
        window._refresh_muon_sites_summary()
        assert panel._table.rowCount() == 2
        assert renders == []

        window._set_active_route("muon_sites")
        qapp.processEvents()
        assert renders.count(panel._viewer) == 1
    finally:
        window.close()


def test_pin_from_non_active_host_workspace_is_rejected(qapp, tmp_path: Path, monkeypatch) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        # A workspace whose clustering was run against a different host than the
        # active one must not overlay its old fractional sites on the new host.
        other_host = host_structure()
        workspace = _clustered_workspace(other_host, uuid4())
        window._set_relaxation_analysis_workspaces([workspace])
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: None,
        )
        window._on_muon_sites_pin_requested(str(workspace["id"]))
        assert muon_sites_summary_snapshots(doc) == {}
    finally:
        window.close()


def test_pin_snapshot_is_keyed_to_workspace_host(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        workspace = _clustered_workspace(host, uuid4())
        window._set_relaxation_analysis_workspaces([workspace])
        window._on_muon_sites_pin_requested(str(workspace["id"]))
        stored = muon_sites_summary_snapshots(doc)[POSITIVE_MUON_CHARGE_STATE]
        # The snapshot host is the workspace's host (which equals the active host
        # here), so its sites are always interpreted in the correct cell.
        assert stored["host_structure_id"] == str(host.id)
    finally:
        window.close()


def test_pin_with_unsupported_charge_state_is_rejected(qapp, tmp_path: Path, monkeypatch) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        # A charge state with no summary panel must not be stored under a key
        # that can never be rendered or unpinned.
        workspace = _clustered_workspace(host, uuid4(), charge_state="neutral")
        window._set_relaxation_analysis_workspaces([workspace])
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: None,
        )
        window._on_muon_sites_pin_requested(str(workspace["id"]))
        assert muon_sites_summary_snapshots(doc) == {}
    finally:
        window.close()
