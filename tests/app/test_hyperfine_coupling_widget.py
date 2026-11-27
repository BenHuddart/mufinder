"""Tests for the Hyperfine Coupling widget."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from mufinder.app.hyperfine_coupling_widget import (
    RESULT_COLUMNS,
    USE_IN_MAGNETIC_FIELDS_TEXT,
    HyperfineCouplingWidget,
    HyperfineResultRow,
    HyperfineSourceRow,
)
from tests.support.row_menu import (
    ENDED_ROW_STATUSES,
    assert_offers_no_reset,
    install_fake_menu,
    open_row_menu,
    trigger,
)


def _source_row(key: str = "site-a", status: str = "ready", pinned: str = "") -> HyperfineSourceRow:
    return HyperfineSourceRow(
        key=key,
        label="Site A",
        charge_state="Positive muon",
        source_job="relax-a",
        status=status,
        pinned_label=pinned,
    )


def _result_row(key: str = "res-a", reason: str = "") -> HyperfineResultRow:
    return HyperfineResultRow(
        key=key,
        site="Site A",
        charge_state="Positive muon",
        isotropic="-324.21",
        contact_field="-1.1946",
        pseudo_contact_field="-0.9434",
        ratio="1.266",
        dipolar="-137.61, 68.80, 68.81",
        handoff_reason=reason,
    )


def test_results_table_has_the_documented_columns(qapp) -> None:
    widget = HyperfineCouplingWidget()

    headers = [
        widget._results_table.horizontalHeaderItem(column).text()
        for column in range(widget._results_table.columnCount())
    ]

    assert headers == [
        "Site",
        "Charge state",
        "A_iso (MHz)",
        "B_c (T)",
        "B_c pseudo (T)",
        "Reconstructed / pseudo",
        "Dipolar (MHz)",
    ]
    assert headers == [title for title, _tip in RESULT_COLUMNS]
    widget.set_results([_result_row()])
    assert [widget._results_table.item(0, column).text() for column in range(7)] == [
        "Site A",
        "Positive muon",
        "-324.21",
        "-1.1946",
        "-0.9434",
        "1.266",
        "-137.61, 68.80, 68.81",
    ]


def test_use_in_magnetic_fields_needs_one_result_that_can_be_handed_over(qapp) -> None:
    widget = HyperfineCouplingWidget()
    emitted: list[str] = []
    widget.use_in_magnetic_fields_requested.connect(emitted.append)
    button = widget._use_in_magnetic_fields_button
    widget.set_results([_result_row("ok"), _result_row("no-spins", reason="No Mulliken spins.")])

    assert button.text() == USE_IN_MAGNETIC_FIELDS_TEXT
    assert not button.isEnabled()
    assert button.toolTip() == "Select one completed result."

    widget._results_table.selectRow(1)
    assert not button.isEnabled()
    assert button.toolTip() == "No Mulliken spins."

    widget._results_table.selectRow(0)
    assert button.isEnabled()
    button.click()
    assert emitted == ["ok"]


def test_source_rows_rename_in_place_unless_pinned(qapp) -> None:
    widget = HyperfineCouplingWidget()
    renamed: list[tuple[str, str]] = []
    widget.source_label_renamed.connect(lambda key, label: renamed.append((key, label)))
    widget.set_site_scope(show_every_site=True)
    widget.set_sources([_source_row("a"), _source_row("b", pinned="Mu1")])

    assert widget._source_label.text() == "Eligible relaxed sites (2)"
    editable = widget._sources_table.item(0, 0)
    pinned = widget._sources_table.item(1, 0)
    assert editable.flags() & Qt.ItemFlag.ItemIsEditable
    assert not pinned.flags() & Qt.ItemFlag.ItemIsEditable
    assert pinned.text() == "Mu1"
    editable.setText("Renamed")
    assert renamed == [("a", "Renamed")]


def test_the_page_has_no_reset_button(qapp) -> None:
    widget = HyperfineCouplingWidget()
    widget.set_sources([_source_row("a", status="failed")])

    texts = [button.text() for button in widget.findChildren(QPushButton)]

    assert not [text for text in texts if "reset" in text.casefold()]


@pytest.mark.parametrize("status", ENDED_ROW_STATUSES)
def test_an_ended_row_menu_runs_and_configures_without_reset(qapp, monkeypatch, status) -> None:
    widget = HyperfineCouplingWidget()
    widget.set_sources([_source_row("a", status=status), _source_row("b")])
    submitted: list[object] = []
    configured: list[object] = []
    widget.submit_cluster_requested.connect(submitted.append)
    widget.configure_requested.connect(configured.append)
    install_fake_menu(monkeypatch)

    menu = open_row_menu(widget._sources_table, row=0)

    assert_offers_no_reset(menu)
    # Rows come from completed relaxations, so there is no Remove.
    assert "Remove" not in menu.texts()
    trigger(menu, "Run on Cluster")
    trigger(menu, "Configure…")
    assert submitted == [["a"]]
    assert configured == [["a"]]
