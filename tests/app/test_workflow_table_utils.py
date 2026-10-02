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

from mufinder.app.workflow_table_utils import confirm_destructive as _real_confirm_destructive
from mufinder.app import workflow_table_utils
from mufinder.app.workflow_table_utils import (
    RESET_FAILED_ACTION_TEXT,
    RESET_FAILED_TABLE_TOOLTIP,
    RESETTABLE_ROW_STATUSES,
    RowMenuAction,
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


def test_selection_supports_reset_rejects_a_partially_keyed_selection() -> None:
    # Two selected failed rows but only one key: the second row is keyless, and
    # enabling here would reset only the first while the batch cleanup prompt
    # claimed to cover both.
    assert selection_supports_reset(["a"], ["failed", "failed"]) is False
    assert selection_supports_reset(["a", "b"], ["failed", "failed"]) is True


def test_bind_reset_failed_context_menu_disables_reset_when_a_selected_row_has_no_key(
    qapp, monkeypatch
) -> None:
    install_fake_menu(monkeypatch)
    table = create_workflow_table(("Label", "Status"))
    for row, (label, status) in enumerate(((" row_0 ", "failed"), ("", "failed"))):
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(label))
        table.setItem(row, 1, QTableWidgetItem(status))
    emitted: list[list[object]] = []
    bind_reset_failed_context_menu(
        table,
        # Mirrors the real widgets: an empty label yields no submission key.
        key_for_row=lambda row: ((table.item(row, 0).text().strip() or None) if table.item(row, 0) else None),
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        emit_reset=emitted.append,
    )
    table.resize(200, 200)
    table.selectAll()

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert menu.reset_action().enabled is False
    assert emitted == []


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


def test_bind_reset_failed_context_menu_advertises_itself_via_tooltip(qapp) -> None:
    # A one-entry menu, greyed out except on failed rows, is invisible without
    # this: nothing on screen tells a user with a stuck row to right-click.
    table, _ = _reset_menu_table(qapp, ("failed",))
    assert table.toolTip() == RESET_FAILED_TABLE_TOOLTIP


def test_bind_reset_failed_context_menu_preserves_a_callers_own_tooltip(qapp) -> None:
    table = create_workflow_table(("Label", "Status"))
    table.setToolTip("widget-specific guidance")
    bind_reset_failed_context_menu(
        table,
        key_for_row=lambda _row: "a",
        status_for_row=lambda _row: "failed",
        emit_reset=lambda _keys: None,
    )
    assert table.toolTip() == "widget-specific guidance"


# ---------------------------------------------------------------------------
# RowMenuAction / additional_actions — the "Continue relaxation…" extension
# ---------------------------------------------------------------------------


def test_bind_reset_failed_context_menu_default_additional_actions_adds_nothing(
    qapp, monkeypatch
) -> None:
    """The other five tables call ``bind_reset_failed_context_menu`` with no
    ``additional_actions``, so their menu must stay exactly one action."""

    install_fake_menu(monkeypatch)
    table, _emitted = _reset_menu_table(qapp, ("failed",))

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert [action.text for action in menu.actions] == [RESET_FAILED_ACTION_TEXT]


def _reset_menu_table_with_continue_action(
    qapp,
    statuses: tuple[str, ...],
    *,
    is_enabled,
):
    table = create_workflow_table(("Label", "Status"))
    for row, status in enumerate(statuses):
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(f"row_{row}"))
        table.setItem(row, 1, QTableWidgetItem(status))
    reset_emitted: list[list[object]] = []
    continue_emitted: list[list[object]] = []
    bind_reset_failed_context_menu(
        table,
        key_for_row=lambda row: (item.text() if (item := table.item(row, 0)) else None),
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        emit_reset=reset_emitted.append,
        additional_actions=(
            RowMenuAction(
                text="Continue relaxation…",
                is_enabled=is_enabled,
                trigger=continue_emitted.append,
            ),
        ),
    )
    table.resize(200, 200)
    return table, reset_emitted, continue_emitted


def test_open_reset_failed_context_menu_appends_additional_action_after_reset(
    qapp, monkeypatch
) -> None:
    install_fake_menu(monkeypatch)
    table, _reset_emitted, _continue_emitted = _reset_menu_table_with_continue_action(
        qapp, ("completed",), is_enabled=lambda _keys: True
    )

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert [action.text for action in menu.actions] == [
        RESET_FAILED_ACTION_TEXT,
        "Continue relaxation…",
    ]


def test_additional_action_enabled_state_is_driven_by_its_own_predicate(
    qapp, monkeypatch
) -> None:
    # A completed row is never reset-eligible, but "Continue relaxation…" can
    # still be enabled for it — the two actions gate independently.
    install_fake_menu(monkeypatch)
    table, _reset_emitted, _continue_emitted = _reset_menu_table_with_continue_action(
        qapp, ("completed",), is_enabled=lambda keys: len(keys) == 1
    )

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert menu.reset_action().enabled is False
    assert menu.action("Continue relaxation…").enabled is True


def test_additional_action_trigger_receives_the_resolved_keys_and_fires_independently(
    qapp, monkeypatch
) -> None:
    install_fake_menu(monkeypatch)
    table, reset_emitted, continue_emitted = _reset_menu_table_with_continue_action(
        qapp, ("completed",), is_enabled=lambda keys: len(keys) == 1
    )

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    action.triggered.emit()

    assert continue_emitted == [["row_0"]]
    # The (disabled, absent) reset action never fired from the same click.
    assert reset_emitted == []


def test_additional_action_disabled_when_multiple_rows_selected(qapp, monkeypatch) -> None:
    install_fake_menu(monkeypatch)
    table, _reset_emitted, _continue_emitted = _reset_menu_table_with_continue_action(
        qapp, ("completed", "completed"), is_enabled=lambda keys: len(keys) == 1
    )
    table.selectAll()

    menu = open_reset_menu(table, row=0)

    assert menu is not None
    assert menu.action("Continue relaxation…").enabled is False


# ---------------------------------------------------------------------------
# Destructive-action confirmation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("click_accept", "expected"), [(True, True), (False, False)])
def test_confirm_destructive_maps_buttons_and_defaults_to_the_safe_choice(
    qapp, monkeypatch, click_accept, expected
) -> None:
    from PySide6.QtWidgets import QMessageBox, QWidget

    seen: dict[str, object] = {}

    def _fake_exec(box):
        labels = {button.text(): button for button in box.buttons()}
        seen["labels"] = sorted(labels)
        seen["default"] = box.defaultButton().text()
        seen["escape"] = box.escapeButton().text()
        # The window title is not asserted: macOS message boxes do not show one.
        seen["text"] = (box.text(), box.informativeText())
        chosen = labels["Cancel 2 jobs" if click_accept else "Keep running"]
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda _self: chosen)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec)

    parent = QWidget()
    result = _real_confirm_destructive(
        parent, "Cancel jobs", "Cancel 2 jobs?", "They are running.", "Cancel 2 jobs", "Keep running"
    )

    assert result is expected
    assert seen["labels"] == ["Cancel 2 jobs", "Keep running"]
    assert seen["default"] == "Keep running"
    assert seen["escape"] == "Keep running"
    assert seen["text"] == ("Cancel 2 jobs?", "They are running.")


def test_confirm_row_removal_states_count_and_what_is_at_stake(
    qapp, confirmed_destructive_actions
) -> None:
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    assert workflow_table_utils.confirm_row_removal(parent, 1) is True
    assert workflow_table_utils.confirm_row_removal(
        parent, 4, noun=("path", "paths"), submitted_count=3, cancelling_count=1
    ) is True

    assert confirmed_destructive_actions == [
        {
            "title": "Remove row",
            "text": "Remove 1 row?",
            "detail": "",
            "accept_label": "Remove 1 row",
            "reject_label": "Cancel",
        },
        {
            "title": "Remove paths",
            "text": "Remove 4 paths?",
            "detail": (
                "1 job is queued or running and will be cancelled. "
                "2 paths already have a submitted job."
            ),
            "accept_label": "Remove 4 paths",
            "reject_label": "Cancel",
        },
    ]


def test_confirm_row_removal_of_nothing_is_declined_without_asking(
    qapp, confirmed_destructive_actions
) -> None:
    from PySide6.QtWidgets import QWidget

    assert workflow_table_utils.confirm_row_removal(QWidget(), 0) is False
    assert confirmed_destructive_actions == []
