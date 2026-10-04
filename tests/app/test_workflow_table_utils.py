"""Regression tests for shared workflow table/button helpers.

The button tests assert the exact rendered text/tooltip for every wording
variant in use so that adopting the helper across widgets cannot silently
drift the visible text. The table tests pin down ``create_workflow_table``'s
default chrome (the DL-4 house style) plus every keyword-only option added
for the raw-``QTableWidget`` conversion sites, so future edits cannot
silently change the shared defaults out from under existing callers.
"""

from __future__ import annotations

import sys

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


def _resettable(statuses: list[str], keys: list[object | None]) -> list[object]:
    return workflow_table_utils.resettable_keys(
        range(len(statuses)),
        key_for_row=lambda row: keys[row],
        status_for_row=lambda row: statuses[row],
    )


@pytest.mark.parametrize("status", sorted(RESETTABLE_ROW_STATUSES))
def test_resettable_keys_accepts_every_resettable_status(status: str) -> None:
    assert _resettable([status], ["a"]) == ["a"]


def test_resettable_keys_keeps_only_rows_a_reset_applies_to() -> None:
    assert _resettable([], []) == []
    assert _resettable(["completed"], ["a"]) == []
    # A mixed selection resets the rows it can and skips the rest.
    assert _resettable(["failed", "running", "cancelled"], ["a", "b", "c"]) == ["a", "c"]
    # A row with no key (unlabeled) has no job to reset.
    assert _resettable(["failed", "failed"], ["a", None]) == ["a"]
    assert _resettable(["failed"], [None]) == []


@pytest.mark.parametrize(
    ("eligible", "selected", "expected"),
    [(0, 3, ""), (3, 3, ""), (2, 5, " (2 of 5)"), (1, 2, " (1 of 2)"), (0, 0, "")],
)
def test_partial_count_suffix_appears_only_for_a_mixed_selection(eligible, selected, expected) -> None:
    assert workflow_table_utils.partial_count_suffix(eligible, selected) == expected


def test_bind_reset_failed_context_menu_resets_keyed_rows_and_counts_the_keyless_one(
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

    # The keyless row is skipped, and the entry says so before the reset prompt
    # (which names the number of jobs) is shown.
    assert menu.reset_action().text == "Reset Failed Jobs... (1 of 2)"
    assert menu.reset_action().enabled is True
    assert emitted == [["row_0"]]


def test_bind_reset_failed_context_menu_resets_the_eligible_rows_of_a_mixed_selection(
    qapp, monkeypatch
) -> None:
    install_fake_menu(monkeypatch)
    table, emitted = _reset_menu_table(qapp, ("failed", "running", "cancelled", "completed"))
    table.selectAll()

    menu = open_reset_menu(table, row=0)

    assert menu.reset_action().text == "Reset Failed Jobs... (2 of 4)"
    assert emitted == [["row_0", "row_2"]]


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


# ---------------------------------------------------------------------------
# Shared row menu
# ---------------------------------------------------------------------------


def _status_table(qapp, statuses: list[str]):
    table = create_workflow_table(("Label", "Status"))
    table.setRowCount(len(statuses))
    for row, status in enumerate(statuses):
        table.setItem(row, 0, QTableWidgetItem(f"row{row}"))
        table.setItem(row, 1, QTableWidgetItem(status))
    return table


def _bind_standard_menu(table, calls, *, emit_reset=True, **omit):
    def record(name):
        return lambda rows: calls.append((name, list(rows)))

    callbacks = {name: record(name) for name in ("run_local", "run_cluster", "cancel", "inspect", "configure", "remove")}
    for name in omit:
        callbacks[name] = None
    leading, trailing = workflow_table_utils.standard_row_actions(
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        confirm_parent=table,
        **callbacks,
    )
    bind_reset_failed_context_menu(
        table,
        key_for_row=lambda row: table.item(row, 0).text(),
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        emit_reset=(lambda keys: calls.append(("reset", list(keys)))) if emit_reset else None,
        leading_actions=leading,
        additional_actions=trailing,
    )


def test_standard_row_menu_has_one_order_with_reset_in_its_place(qapp, monkeypatch) -> None:
    table = _status_table(qapp, ["ready"])
    _bind_standard_menu(table, [])
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(table)

    assert [action.text for action in menu.actions] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Inspect Failure",
        "Reset Failed Jobs...",
        "Configure...",
        "Remove",
    ]


def test_standard_row_menu_omits_actions_a_table_does_not_have(qapp, monkeypatch) -> None:
    table = _status_table(qapp, ["ready"])
    _bind_standard_menu(table, [], emit_reset=False, remove=None, inspect=None)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(table)

    assert [action.text for action in menu.actions] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Configure...",
    ]
    # Without a reset action there is nothing for the reset tooltip to advertise.
    assert table.toolTip() == ""


@pytest.mark.parametrize(
    ("statuses", "enabled"),
    [
        (["ready", "not generated"], {"Run Local", "Run on Cluster", "Configure...", "Remove"}),
        (["running", "queued"], {"Cancel", "Configure...", "Remove"}),
        (["failed", "stopped"], {"Inspect Failure", "Reset Failed Jobs...", "Configure...", "Remove"}),
        (["completed", "completed"], {"Configure...", "Remove"}),
    ],
)
def test_standard_row_menu_enables_actions_from_row_status(qapp, monkeypatch, statuses, enabled) -> None:
    table = _status_table(qapp, statuses)
    _bind_standard_menu(table, [])
    install_fake_menu(monkeypatch)
    table.selectAll()

    menu = open_reset_menu(table)

    assert {action.text for action in menu.actions if action.enabled} == enabled


def test_standard_row_menu_offers_actions_for_the_eligible_rows_of_a_mixed_selection(
    qapp, monkeypatch, confirmed_destructive_actions
) -> None:
    table = _status_table(qapp, ["ready", "running", "failed", "running"])
    calls: list[tuple[str, list[object]]] = []
    _bind_standard_menu(table, calls)
    install_fake_menu(monkeypatch)
    table.selectAll()

    menu = open_reset_menu(table)

    assert [action.text for action in menu.actions] == [
        "Run Local (1 of 4)",
        "Run on Cluster (1 of 4)",
        "Cancel (2 of 4)",
        "Inspect Failure (1 of 4)",
        "Reset Failed Jobs... (1 of 4)",
        "Configure...",
        "Remove",
    ]
    assert all(action.enabled for action in menu.actions)

    menu.action("Cancel (2 of 4)").triggered.emit()
    menu.action("Inspect Failure (1 of 4)").triggered.emit()
    menu.action("Configure...").triggered.emit()

    # The fake menu's exec() already clicked the enabled reset entry.
    assert calls == [
        ("reset", ["row2"]),
        ("cancel", [1, 3]),
        ("inspect", [2]),
        ("configure", [0, 1, 2, 3]),
    ]
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Cancel 2 calculations?"]


def test_standard_row_menu_passes_selected_rows_and_confirms_cancel(
    qapp, monkeypatch, confirmed_destructive_actions
) -> None:
    table = _status_table(qapp, ["running", "running"])
    calls: list[tuple[str, list[int]]] = []
    _bind_standard_menu(table, calls)
    install_fake_menu(monkeypatch)
    table.selectAll()

    menu = open_reset_menu(table)
    menu.action("Configure...").triggered.emit()
    menu.action("Cancel").triggered.emit()
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    menu.action("Cancel").triggered.emit()

    assert calls == [("configure", [0, 1]), ("cancel", [0, 1])]
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Cancel 2 calculations?"]


def test_row_menu_action_separator_is_drawn_above_it(qapp, monkeypatch) -> None:
    table = _status_table(qapp, ["ready"])
    bind_reset_failed_context_menu(
        table,
        key_for_row=lambda row: table.item(row, 0).text(),
        status_for_row=lambda row: workflow_row_status_text(table, row, 1),
        emit_reset=lambda keys: None,
        additional_actions=(
            RowMenuAction("Extra", lambda keys: True, lambda keys: None, separator_before=True),
        ),
    )
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(table)

    assert [action.text for action in menu.actions] == ["Reset Failed Jobs...", "Extra"]
    assert menu.separators_before == [1]


# ---------------------------------------------------------------------------
# Selection feedback
# ---------------------------------------------------------------------------


def _feedback_table(qapp, rows: int = 3):
    from PySide6.QtWidgets import QLabel

    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    table = create_workflow_table(("Label", "Action", "Solo"))
    table.setRowCount(rows)
    for row in range(rows):
        table.setItem(row, 0, QTableWidgetItem(f"row{row}"))
        table.setCellWidget(row, 1, SelectionSnapshotPushButton("Go", table))
        solo = SelectionSnapshotPushButton("Solo", table)
        solo.propagates_to_selection = False
        table.setCellWidget(row, 2, solo)
    widgets = {
        "count": QLabel(),
        "hint": QLabel(),
        "local": QPushButton(),
        "cluster": QPushButton(),
        "remove": QPushButton(),
    }
    return table, widgets


def _select(table, rows) -> None:
    from PySide6.QtCore import QItemSelectionModel

    model = table.selectionModel()
    model.clearSelection()
    for row in rows:
        model.select(
            table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_selection_feedback_tracks_count_hint_run_labels_and_buttons(qapp) -> None:
    table, widgets = _feedback_table(qapp)
    workflow_table_utils.SelectionFeedback(
        table,
        count_label=widgets["count"],
        multi_row_hint=widgets["hint"],
        multi_row_hint_text=lambda count: f"{count} rows",
        run_local_button=widgets["local"],
        run_cluster_button=widgets["cluster"],
        selection_buttons=(widgets["remove"],),
    )

    assert widgets["count"].text() == ""
    assert widgets["hint"].isHidden()
    assert widgets["local"].text() == "Run All Locally"
    assert widgets["cluster"].text() == "Run All on Cluster"
    assert widgets["local"].isEnabled()
    assert not widgets["remove"].isEnabled()

    _select(table, [1])
    assert widgets["count"].text() == "1 selected"
    assert widgets["hint"].isHidden()
    assert widgets["local"].text() == "Run Selected Locally"
    assert widgets["cluster"].text() == "Run Selected on Cluster"
    assert widgets["remove"].isEnabled()

    _select(table, [0, 2])
    assert widgets["count"].text() == "2 selected"
    assert not widgets["hint"].isHidden()
    assert widgets["hint"].text() == "2 rows"

    table.setRowCount(0)
    assert widgets["count"].text() == ""
    assert not widgets["local"].isEnabled()
    assert not widgets["cluster"].isEnabled()


def test_selection_feedback_marks_only_propagating_controls_in_a_multi_row_selection(qapp) -> None:
    table, _widgets = _feedback_table(qapp)
    workflow_table_utils.SelectionFeedback(table)
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY

    def marked(column: int) -> list[bool]:
        return [bool(table.cellWidget(row, column).property(prop)) for row in range(3)]

    _select(table, [1])
    assert marked(1) == [False, False, False]

    _select(table, [0, 2])
    assert marked(1) == [True, False, True]
    assert marked(2) == [False, False, False]

    _select(table, [])
    assert marked(1) == [False, False, False]


def test_selection_feedback_only_the_active_table_drives_shared_controls(qapp) -> None:
    first, widgets = _feedback_table(qapp)
    second, _unused = _feedback_table(qapp)
    active = {"table": first}
    feedbacks = [
        workflow_table_utils.SelectionFeedback(
            table,
            count_label=widgets["count"],
            is_active=lambda bound=table: active["table"] is bound,
        )
        for table in (first, second)
    ]

    _select(second, [0, 1])
    assert widgets["count"].text() == ""
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    assert bool(second.cellWidget(0, 1).property(prop)) is True

    active["table"] = second
    for feedback in feedbacks:
        feedback.refresh()
    assert widgets["count"].text() == "2 selected"


def test_selection_feedback_survives_the_table_being_deleted(qapp, monkeypatch) -> None:
    import shiboken6

    # PySide reports an exception raised in a slot through sys.excepthook
    # rather than propagating it, so capture it there.
    slot_errors: list[BaseException] = []
    monkeypatch.setattr(sys, "excepthook", lambda _type, value, _tb: slot_errors.append(value))
    table, widgets = _feedback_table(qapp)
    feedback = workflow_table_utils.SelectionFeedback(table, count_label=widgets["count"])
    _select(table, [0, 1])

    # Destroying the table tears its model down first, which emits
    # ``modelReset`` while the table's C++ object is already gone (app quit).
    shiboken6.delete(table)
    assert not shiboken6.isValid(table)
    feedback.refresh()

    assert slot_errors == []
    assert widgets["count"].text() == "2 selected"


def test_multi_row_outcome_text_reports_skipped_rows() -> None:
    assert workflow_table_utils.multi_row_outcome_text("Supercell set on", 3, 3) == (
        "Supercell set on 3 rows"
    )
    assert workflow_table_utils.multi_row_outcome_text("Supercell set on", 1, 3) == (
        "Supercell set on 1 row, 2 rows skipped"
    )


def test_show_status_message_uses_the_enclosing_main_window_only(qapp) -> None:
    from PySide6.QtWidgets import QMainWindow, QStatusBar, QWidget

    window = QMainWindow()
    window.setStatusBar(QStatusBar(window))
    child = QWidget()
    window.setCentralWidget(child)
    workflow_table_utils.show_status_message(child, "Done")
    assert window.statusBar().currentMessage() == "Done"

    bare_window = QMainWindow()
    bare_child = QWidget()
    bare_window.setCentralWidget(bare_child)
    workflow_table_utils.show_status_message(bare_child, "Done")
    assert bare_window.findChild(QStatusBar) is None

    # A widget outside any main window is simply ignored.
    workflow_table_utils.show_status_message(QWidget(), "Done")
