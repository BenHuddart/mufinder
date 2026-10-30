"""Behaviour tests for downstream workflows defaulting to the pinned Muon Sites summary."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QDialog, QPushButton

from mufinder.app.magnetic_site_dialog import MuonSiteSelectionDialog
from mufinder.app.pinned_sites_scope import (
    GO_TO_MUON_SITES_TEXT,
    SHOW_EVERY_RELAXED_SITE_TEXT,
    PinnedSitesScopeBar,
)
from mufinder.services.muon_sites_summary import (
    ENERGY_BARRIERS_SCOPE,
    MAGNETIC_FIELDS_SCOPE,
    NO_PINNED_SITES_NOTICE,
    QUANTUM_CORRECTIONS_SCOPE,
    build_muon_sites_summary_snapshot,
    show_every_relaxed_site,
    store_muon_sites_summary_snapshot,
)
from tests.support.main_window import window_with_host
from tests.support.muon_sites import (
    ClusterSpec,
    add_relaxed_muon_site,
    clustered_workspace,
    relaxation_job_id,
    store_workspaces,
)

_window_with_host = partial(window_with_host, filename="pins.mufinder", name="Pins")


def _relaxed_copies(doc, host):
    """``low_a`` and ``low_b`` are relaxed copies of one site; ``high`` is another site."""

    low_a = add_relaxed_muon_site(doc, label="low_a", total_energy=-100.0)
    low_b = add_relaxed_muon_site(doc, label="low_b", total_energy=-99.999)
    high = add_relaxed_muon_site(doc, label="high", total_energy=-99.5)
    workspace = clustered_workspace(
        host,
        [
            ClusterSpec(1, relaxation_job_id(doc, low_a), (0.25, 0.25, 0.25), 0.0, "8c"),
            ClusterSpec(2, relaxation_job_id(doc, high), (0.5, 0.0, 0.0), 0.5, "4b"),
        ],
        name="Positive analysis",
    )
    store_workspaces(doc, workspace)
    return low_a, low_b, high, workspace


def _pin(doc, workspace) -> None:
    snapshot = build_muon_sites_summary_snapshot(workspace)
    assert snapshot is not None
    store_muon_sites_summary_snapshot(doc, str(snapshot["charge_state"]), snapshot)


def _scope_checkbox(widget) -> QCheckBox:
    return next(
        box for box in widget.findChildren(QCheckBox) if box.text() == SHOW_EVERY_RELAXED_SITE_TEXT
    )


def _muon_sites_link(widget) -> QPushButton:
    return next(
        button
        for button in widget.findChildren(QPushButton)
        if button.text() == GO_TO_MUON_SITES_TEXT
    )


def _quantum_site_labels(window) -> list[str]:
    table = window._quantum_corrections_view._sources_table
    return [table.item(row, 0).text() for row in range(table.rowCount())]


# --- scope bar ------------------------------------------------------------------


def test_scope_bar_shows_the_notice_only_for_the_pinned_list(qapp) -> None:
    bar = PinnedSitesScopeBar()
    toggles: list[bool] = []
    bar.show_every_site_toggled.connect(toggles.append)
    try:
        bar.set_state(show_every_site=False, notice="Nothing pinned.")
        assert bar.notice_text() == "Nothing pinned."
        bar.set_state(show_every_site=True, notice="Nothing pinned.")
        assert bar.notice_text() == ""
        assert bar.show_every_site()
        # Restoring state does not echo a toggle back to the controller.
        assert toggles == []
        _scope_checkbox(bar).click()
        assert toggles == [False]
    finally:
        bar.deleteLater()


# --- quantum corrections ------------------------------------------------------


def test_quantum_corrections_lists_the_pinned_sites_by_default(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _relaxed_copies(doc, host)
        _pin(doc, doc.ui_state["relaxation_analysis_workspaces"][0])

        window._refresh_quantum_corrections_view()

        assert _quantum_site_labels(window) == ["8c · cluster 1", "4b · cluster 2"]
        assert window._quantum_corrections_view._source_label.text() == "Pinned muon sites (2)"
    finally:
        window.close()


def test_quantum_corrections_show_every_relaxed_site_toggle_persists(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _relaxed_copies(doc, host)
        _pin(doc, doc.ui_state["relaxation_analysis_workspaces"][0])
        window._refresh_quantum_corrections_view()

        _scope_checkbox(window._quantum_corrections_view).click()

        assert show_every_relaxed_site(doc, QUANTUM_CORRECTIONS_SCOPE)
        assert sorted(_quantum_site_labels(window)) == ["high", "low_a", "low_b"]
        assert window._quantum_corrections_view._source_label.text() == "Eligible relaxed sites (3)"

        _scope_checkbox(window._quantum_corrections_view).click()

        assert not show_every_relaxed_site(doc, QUANTUM_CORRECTIONS_SCOPE)
        assert _quantum_site_labels(window) == ["8c · cluster 1", "4b · cluster 2"]
    finally:
        window.close()


def test_quantum_corrections_without_a_pin_links_to_the_muon_sites_page(
    qapp, tmp_path: Path
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _relaxed_copies(doc, host)
        window._refresh_quantum_corrections_view()

        assert _quantum_site_labels(window) == []
        assert window._quantum_corrections_view._site_scope.notice_text() == NO_PINNED_SITES_NOTICE

        _muon_sites_link(window._quantum_corrections_view).click()

        assert window._main_stack.currentWidget() is window._muon_sites_summary_view
    finally:
        window.close()


def test_pinning_refreshes_the_quantum_corrections_list(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _low_a, _low_b, _high, workspace = _relaxed_copies(doc, host)
        window._refresh_quantum_corrections_view()
        assert _quantum_site_labels(window) == []

        window._on_muon_sites_pin_requested(str(workspace["id"]))

        assert _quantum_site_labels(window) == ["8c · cluster 1", "4b · cluster 2"]
    finally:
        window.close()


def test_a_configured_unpinned_site_stays_in_the_quantum_list(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _low_a, low_b, _high, workspace = _relaxed_copies(doc, host)
        _pin(doc, workspace)
        window._quantum_corrections_controller.ensure_definition([str(low_b.id)])
        definition = window._quantum_corrections_controller.active_definition()
        definition.parameters.setdefault("configuration", {})["row_settings"] = {
            str(low_b.id): {"params": {}, "advanced_params": {}, "extras": {}}
        }

        window._refresh_quantum_corrections_view()

        assert _quantum_site_labels(window) == ["8c · cluster 1", "4b · cluster 2", "low_b"]
    finally:
        window.close()


# --- energy barriers ----------------------------------------------------------


def test_energy_barrier_endpoints_default_to_the_pinned_sites(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        low_a, _low_b, high, workspace = _relaxed_copies(doc, host)
        _pin(doc, workspace)
        view = window._ensure_energy_barriers_view()
        window._refresh_energy_barriers_view()

        combo = view._reactant_combo
        assert [combo.itemData(index) for index in range(combo.count())] == [
            str(low_a.id),
            str(high.id),
        ]
        assert combo.itemText(0).startswith("8c · cluster 1")

        _scope_checkbox(view).click()

        assert show_every_relaxed_site(doc, ENERGY_BARRIERS_SCOPE)
        assert view._reactant_combo.count() == 3
    finally:
        window.close()


def test_energy_barrier_endpoints_keep_the_pinned_pair_when_widened(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        low_a, low_b, high, workspace = _relaxed_copies(doc, host)
        _pin(doc, workspace)
        # A pair stored before pinning names an unpinned relaxed copy.
        doc.ui_state["energy_barriers_form"] = {
            "selected_reactant_key": str(low_b.id),
            "selected_product_key": str(high.id),
        }
        view = window._ensure_energy_barriers_view()
        window._refresh_energy_barriers_view()

        assert view.selected_endpoint_keys() == (str(low_a.id), str(high.id))
        assert doc.ui_state["energy_barriers_form"]["selected_reactant_key"] == str(low_a.id)

        _scope_checkbox(view).click()

        assert view.selected_endpoint_keys() == (str(low_a.id), str(high.id))
    finally:
        window.close()


def test_energy_barriers_without_a_pin_links_to_the_muon_sites_page(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _relaxed_copies(doc, host)
        view = window._ensure_energy_barriers_view()
        window._refresh_energy_barriers_view()

        assert view._reactant_combo.count() == 0
        assert view._site_scope.notice_text() == NO_PINNED_SITES_NOTICE

        _muon_sites_link(view).click()

        assert window._main_stack.currentWidget() is window._muon_sites_summary_view
    finally:
        window.close()


# --- magnetic fields ----------------------------------------------------------


def _relaxation_rows(dialog) -> list[tuple[str, str]]:
    table = dialog._site_table
    return [
        (table.item(row, 0).text(), table.item(row, 2).text()) for row in range(table.rowCount())
    ]


def test_magnetic_site_dialog_lists_pinned_clusters_by_default(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _low_a, _low_b, _high, workspace = _relaxed_copies(doc, host)
        unpinned = clustered_workspace(
            host, [ClusterSpec(1, relaxation_job_id(doc, _low_b))], name="Other analysis"
        )
        store_workspaces(doc, workspace, unpinned)
        _pin(doc, workspace)
        dialog = MuonSiteSelectionDialog(doc, host)
        try:
            dialog._site_source_combo.setCurrentIndex(
                dialog._site_source_combo.findData("relaxation_clustering")
            )
            assert _relaxation_rows(dialog) == [
                ("Positive analysis cluster 1", "8c · cluster 1"),
                ("Positive analysis cluster 2", "4b · cluster 2"),
            ]

            _scope_checkbox(dialog).click()

            assert show_every_relaxed_site(doc, MAGNETIC_FIELDS_SCOPE)
            assert len(_relaxation_rows(dialog)) == 3
        finally:
            dialog.deleteLater()
    finally:
        window.close()


def test_magnetic_site_dialog_matches_pins_by_representative_after_reclustering(
    qapp, tmp_path: Path
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        low_a, _low_b, high, workspace = _relaxed_copies(doc, host)
        _pin(doc, workspace)
        # Re-clustering without re-pinning renumbers the clusters.
        reclustered = clustered_workspace(
            host,
            [
                ClusterSpec(1, relaxation_job_id(doc, high), (0.5, 0.0, 0.0), 0.0),
                ClusterSpec(2, relaxation_job_id(doc, low_a), (0.25, 0.25, 0.25), 0.5),
            ],
            name="Positive analysis",
        )
        reclustered["id"] = workspace["id"]
        store_workspaces(doc, reclustered)
        dialog = MuonSiteSelectionDialog(doc, host)
        try:
            dialog._site_source_combo.setCurrentIndex(
                dialog._site_source_combo.findData("relaxation_clustering")
            )
            options = dialog._site_options
            assert [(option.cluster_label, option.fractional_position) for option in options] == [
                ("8c · cluster 1", (0.25, 0.25, 0.25)),
                ("4b · cluster 2", (0.5, 0.0, 0.0)),
            ]
        finally:
            dialog.deleteLater()
    finally:
        window.close()


def test_magnetic_site_dialog_without_a_pin_links_to_the_muon_sites_page(
    qapp, tmp_path: Path
) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _relaxed_copies(doc, host)
        dialog = MuonSiteSelectionDialog(doc, host)
        requested: list[bool] = []
        dialog.open_muon_sites_requested.connect(lambda: requested.append(True))
        try:
            dialog._site_source_combo.setCurrentIndex(
                dialog._site_source_combo.findData("relaxation_clustering")
            )
            assert _relaxation_rows(dialog) == []
            assert dialog._site_scope.notice_text() == NO_PINNED_SITES_NOTICE

            _muon_sites_link(dialog).click()

            assert requested == [True]
            assert dialog.result() == QDialog.DialogCode.Rejected
        finally:
            dialog.deleteLater()
    finally:
        window.close()


def test_magnetic_fields_default_to_the_lowest_energy_pinned_site(qapp, tmp_path: Path) -> None:
    window, _service, doc, host = _window_with_host(tmp_path)
    try:
        _low_a, _low_b, _high, workspace = _relaxed_copies(doc, host)
        _pin(doc, workspace)

        window._refresh_magnetic_field_view()

        site = window._magnetic_field_view.selected_muon_site()
        assert site is not None
        assert site.source == "relaxation_clustering"
        assert site.cluster_label == "8c · cluster 1"
    finally:
        window.close()
