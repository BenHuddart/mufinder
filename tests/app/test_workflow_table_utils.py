"""Regression tests for shared workflow table/button helpers.

The button tests assert the exact rendered text/tooltip for every wording
variant in use so that adopting the helper across widgets cannot silently
drift the visible text. The table tests pin down ``create_workflow_table``'s
default chrome (the DL-4 house style) plus every keyword-only option added
for the raw-``QTableWidget`` conversion sites, so future edits cannot
silently change the shared defaults out from under existing callers.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QFrame, QHeaderView, QPushButton, QTableWidgetItem

from mufinder.app.workflow_table_utils import (
    RESETTABLE_ROW_STATUSES,
    bind_reset_failed_context_menu,
    create_workflow_table,
    selection_supports_reset,
    style_workflow_configure_button,
    workflow_row_status_text,
)
from tests.support.reset_menu import install_fake_menu, open_reset_menu


@pytest.fixture()
def button(qapp) -> QPushButton:
    return QPushButton()


def test_create_workflow_table_default_chrome(qapp) -> None:
    table = create_workflow_table(("Alpha", "Beta"))
    assert table.frameShape() == QFrame.Shape.NoFrame
    assert table.wordWrap() is False
    assert table.alternatingRowColors() is True
    assert table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectRows
    assert table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert table.verticalHeader().isHidden() is True
    assert table.horizontalHeader().stretchLastSection() is True
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert table.property("squareTopCorners") == "true"
    assert table.property("tabbedTable") is None


def test_create_workflow_table_resize_to_contents_columns(qapp) -> None:
    table = create_workflow_table(("A", "B", "C"), resize_to_contents_columns=(1,))
    header = table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch


def test_create_workflow_table_selection_overrides(qapp) -> None:
    table = create_workflow_table(
        ("A",),
        selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
        selection_behavior=QAbstractItemView.SelectionBehavior.SelectItems,
    )
    assert table.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection
    assert table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectItems


def test_create_workflow_table_editable(qapp) -> None:
    table = create_workflow_table(("A",), editable=True)
    assert table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers


def test_create_workflow_table_show_vertical_header(qapp) -> None:
    table = create_workflow_table(("A",), show_vertical_header=True)
    assert table.verticalHeader().isHidden() is False


def test_create_workflow_table_tabbed_corners(qapp) -> None:
    table = create_workflow_table(("A",), corner_style="tabbed")
    assert table.property("squareTopCorners") == "true"
    assert table.property("tabbedTable") == "true"


def test_create_workflow_table_stretch_last_section_and_resize_mode(qapp) -> None:
    table = create_workflow_table(
        ("A", "B"),
        stretch_last_section=False,
        default_resize_mode=QHeaderView.ResizeMode.Interactive,
    )
    header = table.horizontalHeader()
    assert header.stretchLastSection() is False
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive


def test_configure_button_energy_barriers_defaults(button: QPushButton) -> None:
    # Adopter baseline: no noun/used → "for this row." with no "used".
    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=False, locked=False
    )
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure Energy Barriers parameters for this row."

    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=True, locked=False
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom Energy Barriers parameters for this row."

    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=False, locked=True
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the Energy Barriers parameters for this row."


def test_configure_button_quantum_corrections_defaults(button: QPushButton) -> None:
    style_workflow_configure_button(
        button, workflow_label="Quantum Corrections", configured=False, locked=True
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the Quantum Corrections parameters for this row."


def test_configure_button_relaxation_wording(button: QPushButton) -> None:
    # Relaxation / muon relaxation: label "CASTEP", noun "row", "used" phrasing.
    kwargs = {"workflow_label": "CASTEP", "noun": "row", "used": True}

    style_workflow_configure_button(button, configured=False, locked=False, **kwargs)
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure CASTEP parameters for this row."

    style_workflow_configure_button(button, configured=True, locked=False, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom CASTEP parameters for this row."

    style_workflow_configure_button(button, configured=False, locked=True, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the CASTEP parameters used for this row."


def test_configure_button_uep_wording(button: QPushButton) -> None:
    kwargs = {"workflow_label": "CASTEP", "noun": "UEP source job", "used": True}

    style_workflow_configure_button(button, configured=False, locked=False, **kwargs)
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure CASTEP parameters for this UEP source job."

    style_workflow_configure_button(button, configured=True, locked=False, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom CASTEP parameters for this UEP source job."

    style_workflow_configure_button(button, configured=False, locked=True, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the CASTEP parameters used for this UEP source job."


def test_configure_button_locked_takes_priority_over_configured(button: QPushButton) -> None:
    style_workflow_configure_button(
        button, workflow_label="CASTEP", configured=True, locked=True, noun="row", used=True
    )
    assert button.text() == "Configured..."
    # Locked wins: inspect phrasing, not the edit phrasing.
    assert button.toolTip() == "Inspect the CASTEP parameters used for this row."


# ---------------------------------------------------------------------------
# Shared per-row "Reset Failed Jobs..." context menu
# ---------------------------------------------------------------------------


def _reset_menu_table(qapp, statuses: tuple[str, ...]):
    """Build a two-column Label/Status table wired to the shared reset menu."""

    table = create_workflow_table(("Label", "Status"))
    for row, status in enumerate(statuses):
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(f"row_{row}"))
        table.setItem(row, 1, QTableWidgetItem(status))
    emitted: list[list[object]] = []
    bind_reset_failed_context_menu(
        table,
        key_for_row=lambda row: (item.text() if (item := table.item(row, 0)) else None),
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        emit_reset=emitted.append,
    )
    table.resize(200, 200)
    return table, emitted


def test_resettable_row_statuses_match_reset_failed_job_contract() -> None:
    # ProjectService.reset_failed_job accepts exactly failed/stopped/cancelled;
    # the menu gate must not drift away from it.
    assert RESETTABLE_ROW_STATUSES == {"failed", "stopped", "cancelled"}


def test_workflow_row_status_text_normalizes_and_tolerates_missing_items(qapp) -> None:
    table = create_workflow_table(("Status",))
    table.insertRow(0)
    table.setItem(0, 0, QTableWidgetItem("  Failed  "))
    table.insertRow(1)
    assert workflow_row_status_text(table, 0, 0) == "failed"
    assert workflow_row_status_text(table, 1, 0) == ""


@pytest.mark.parametrize("status", sorted(RESETTABLE_ROW_STATUSES))
def test_selection_supports_reset_accepts_every_resettable_status(status: str) -> None:
    assert selection_supports_reset(["a"], [status]) is True


def test_selection_supports_reset_rejects_empty_and_mixed_selections() -> None:
    assert selection_supports_reset([], []) is False
    # A row with no key (unlabeled) leaves nothing to reset.
    assert selection_supports_reset([], ["failed"]) is False
    # Mixed selections stay disabled: the cleanup prompt covers the whole batch.
    assert selection_supports_reset(["a", "b"], ["failed", "running"]) is False
    assert selection_supports_reset(["a"], ["completed"]) is False


def test_bind_reset_failed_context_menu_emits_keys_for_failed_row(qapp, monkeypatch) -> None:
    install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("failed",))

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert emitted == [["row_0"]]


def test_bind_reset_failed_context_menu_disables_action_for_running_row(qapp, monkeypatch) -> None:
    install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("running",))

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert menu.reset_action() is not None
    assert menu.reset_action().enabled is False
    assert emitted == []


def test_bind_reset_failed_context_menu_selects_the_clicked_row(qapp, monkeypatch) -> None:
    install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("failed", "failed"))
    table.selectRow(0)

    open_reset_menu(table, row=1)

    # Right-clicking outside the selection retargets it, so the action describes
    # the row under the cursor rather than the stale selection.
    assert emitted == [["row_1"]]


def test_bind_reset_failed_context_menu_keeps_multi_row_selection(qapp, monkeypatch) -> None:
    install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("failed", "cancelled"))
    table.selectAll()

    open_reset_menu(table, row=0)

    assert emitted == [["row_0", "row_1"]]


def test_bind_reset_failed_context_menu_ignores_clicks_below_the_last_row(qapp, monkeypatch) -> None:
    fake_menu = install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("failed",))

    from PySide6.QtCore import QPoint

    table.customContextMenuRequested.emit(QPoint(10, 5000))

    assert fake_menu.instances == []
    assert emitted == []


def test_bind_reset_failed_context_menu_sets_custom_policy(qapp) -> None:
    table, _ = _reset_menu_table(qapp, ("failed",))
    assert table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
