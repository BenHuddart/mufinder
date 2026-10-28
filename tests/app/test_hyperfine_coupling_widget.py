"""Tests for the Hyperfine Coupling widget."""

from __future__ import annotations

from PySide6.QtCore import Qt

from mufinder.app.hyperfine_coupling_widget import (
    RESULT_COLUMNS,
    USE_IN_MAGNETIC_FIELDS_TEXT,
    HyperfineCouplingWidget,
    HyperfineResultRow,
    HyperfineSourceRow,
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


def test_reset_failed_button_follows_the_selection(qapp) -> None:
    widget = HyperfineCouplingWidget()
    resets: list[object] = []
    widget.reset_failed_requested.connect(resets.append)
    widget.set_sources([_source_row("a", status="failed"), _source_row("b")])

    widget._sources_table.selectRow(1)
    assert not widget._reset_failed_button.isEnabled()
    widget._sources_table.selectRow(0)
    assert widget._reset_failed_button.isEnabled()
    widget._reset_failed_button.click()
    assert resets == [["a"]]
