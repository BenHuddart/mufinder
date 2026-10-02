"""Shared helpers for workflow job/result tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from mufinder.app.theme import use_square_top_table_corners, use_tabbed_table_corners

#: Displayed row statuses whose job can be reset back to ``defined`` from the
#: workflow table itself. It mirrors ``ProjectService.reset_failed_job``'s
#: accepted statuses, but as the lowercased strings the Status cell holds.
#:
#: A *dismissed* terminal job is deliberately absent: after
#: :func:`mufinder.app.workflow_submission.effective_row_status` its row already
#: reads ``"ready"``, and its recovery path is the row's own Submit (which
#: performs the same reset lazily). See ``docs/job_lifecycle.rst``.
RESETTABLE_ROW_STATUSES = frozenset({"failed", "stopped", "cancelled"})

#: Menu text for the per-row reset action, shared so every workflow table names
#: the same recovery affordance identically.
RESET_FAILED_ACTION_TEXT = "Reset Failed Jobs..."

#: Table tooltip announcing the reset menu. A one-entry context menu that is
#: greyed out except on failed rows is otherwise invisible — a user with a stuck
#: row has no reason to right-click — so every table that binds the menu also
#: advertises it. Per-item tooltips still take precedence where a widget sets
#: them; this is only the fallback shown over cells that have none.
RESET_FAILED_TABLE_TOOLTIP = (
    "Right-click a failed, stopped, or cancelled row to reset it back to ready for rerun."
)


def bind_delete_shortcuts(
    widget: QWidget,
    slot: Callable[[], None],
    *,
    context: Qt.ShortcutContext = Qt.ShortcutContext.WidgetWithChildrenShortcut,
) -> tuple[QShortcut, QShortcut]:
    """Bind a delete/remove ``slot`` to both Delete and Backspace on ``widget``.

    macOS delivers the physical Delete key as ``Qt.Key.Key_Backspace``, so
    CLAUDE.md requires every delete/remove/cancel/reset shortcut to bind both
    ``QKeySequence.StandardKey.Delete`` and ``Qt.Key.Key_Backspace`` — binding
    only the standard "Delete" key sequence leaves macOS users unable to
    trigger the action from the keyboard. Returns the two ``QShortcut``
    instances so callers can keep references alive (Qt shortcuts are owned by
    their parent widget, but callers historically stash them on ``self`` as
    well).
    """

    delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, widget)
    delete_shortcut.setContext(context)
    delete_shortcut.activated.connect(slot)
    backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), widget)
    backspace_shortcut.setContext(context)
    backspace_shortcut.activated.connect(slot)
    return delete_shortcut, backspace_shortcut


def confirm_destructive(
    parent: QWidget,
    title: str,
    text: str,
    detail: str = "",
    accept_label: str = "Remove",
    reject_label: str = "Cancel",
) -> bool:
    """Ask the user to confirm an action that removes, cancels, or discards something.

    Every destructive action in a job table or the Job Dashboard goes through
    this one dialog, whichever way it was triggered (row control, button, menu,
    or Delete/Backspace), so the question always looks the same. The safe
    choice is the default button and the one Escape maps to. Buttons carry
    explicit labels because "Cancel" is ambiguous when the action being
    confirmed is itself a cancellation.

    Kept as a module-level function so tests replace it instead of driving a
    modal dialog; callers must look it up on the module at call time.
    """

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    if detail:
        box.setInformativeText(detail)
    accept_button = box.addButton(accept_label, QMessageBox.ButtonRole.DestructiveRole)
    reject_button = box.addButton(reject_label, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(reject_button)
    box.setEscapeButton(reject_button)
    box.exec()
    return box.clickedButton() is accept_button


def count_phrase(count: int, noun: tuple[str, str]) -> str:
    """Return ``"1 row"`` / ``"3 rows"`` for a ``(singular, plural)`` noun."""

    return f"{count} {noun[0] if count == 1 else noun[1]}"


def confirm_row_removal(
    parent: QWidget,
    count: int,
    *,
    noun: tuple[str, str] = ("row", "rows"),
    submitted_count: int = 0,
    cancelling_count: int = 0,
) -> bool:
    """Confirm removing ``count`` rows from a workflow table.

    ``submitted_count`` is how many of them already have a job behind them, and
    ``cancelling_count`` how many of those jobs are still queued or running and
    will be cancelled by the removal; both are reported so the prompt says what
    is at stake rather than just how many rows.
    """

    if count <= 0:
        return False
    details: list[str] = []
    if cancelling_count:
        details.append(
            f"{count_phrase(cancelling_count, ('job is', 'jobs are'))} queued or running "
            "and will be cancelled."
        )
    settled = submitted_count - cancelling_count
    if settled > 0:
        details.append(
            f"{count_phrase(settled, noun)} already "
            f"{'has' if settled == 1 else 'have'} a submitted job."
        )
    return confirm_destructive(
        parent,
        f"Remove {noun[0] if count == 1 else noun[1]}",
        f"Remove {count_phrase(count, noun)}?",
        " ".join(details),
        f"Remove {count_phrase(count, noun)}",
    )


def create_workflow_table(
    headers: tuple[str, ...],
    *,
    resize_to_contents_columns: tuple[int, ...] = (),
    selection_mode: QAbstractItemView.SelectionMode = QAbstractItemView.SelectionMode.ExtendedSelection,
    selection_behavior: QAbstractItemView.SelectionBehavior = QAbstractItemView.SelectionBehavior.SelectRows,
    editable: bool = False,
    show_vertical_header: bool = False,
    corner_style: str = "square",
    stretch_last_section: bool = True,
    default_resize_mode: QHeaderView.ResizeMode = QHeaderView.ResizeMode.Stretch,
) -> QTableWidget:
    """Build a table with MuFinder's shared chrome.

    All keyword-only options default to the historical behavior of the
    original ``create_workflow_table`` (no-frame, alternating rows,
    row-selection, no-edit, hidden vertical header, elide-middle, square top
    corners, stretch-to-fill columns), so existing callers are unaffected.

    ``selection_mode``/``selection_behavior`` let sites that need multi-row
    selection (``ExtendedSelection``, the default) opt into single-row or
    no-selection behavior instead. ``editable`` flips the default
    ``NoEditTriggers`` to Qt's standard edit triggers for tables with
    user-editable cells. ``show_vertical_header`` keeps the row-number
    column visible for tables that use it (e.g. as a stable row index).
    ``corner_style`` selects between the default ``"square"`` top-corner
    treatment and ``"tabbed"`` (used inside tab panes). ``stretch_last_section``
    and ``default_resize_mode`` let sites that want every column sized to its
    contents (rather than stretched to fill) opt out of the stretch default.
    """

    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setFrameShape(QFrame.Shape.NoFrame)
    # Long values (file paths, labels) must elide instead of wrapping, so
    # resizeRowsToContents() cannot balloon row heights.
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(selection_behavior)
    table.setSelectionMode(selection_mode)
    if not editable:
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(show_vertical_header)
    table.horizontalHeader().setSectionResizeMode(default_resize_mode)
    for column in resize_to_contents_columns:
        if 0 <= column < len(headers):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(stretch_last_section)
    if corner_style == "tabbed":
        use_tabbed_table_corners(table)
    else:
        use_square_top_table_corners(table)
    return table


def set_workflow_table_item(table: QTableWidget, row: int, column: int, text: str) -> None:
    item = table.item(row, column)
    if item is None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, column, item)
        return
    item.setText(text)


def workflow_row_status_text(table: QTableWidget, row: int, column: int) -> str:
    """Return the normalized status string a workflow table row is displaying.

    Menu gates compare against :data:`RESETTABLE_ROW_STATUSES`, which holds
    lowercased status values, so the cell text is stripped and lowercased here
    rather than at each call site.
    """

    item = table.item(row, column)
    if item is None:
        return ""
    return item.text().strip().lower()


def selection_supports_reset(keys: Sequence[object], statuses: Sequence[str]) -> bool:
    """Return whether the whole selection can be reset as one batch.

    ``statuses`` carries one entry per selected row; ``keys`` carries one entry
    per selected row that yielded a submission key. Both conditions below exist
    because ``_reset_failed_jobs`` prompts once for the entire batch, including
    the destructive output cleanup — so any row the action would silently drop
    makes that prompt misreport what is about to happen:

    - every status must be resettable, which rules out a mixed selection; and
    - ``len(keys)`` must equal ``len(statuses)``, which rules out a selection
      containing a keyless row (an unlabeled or half-built row whose
      ``key_for_row`` returned ``None``). Without this a two-row selection of
      one keyed and one keyless failed row would enable the action and reset
      only the first, leaving the second untouched with no indication why.
    """

    return (
        bool(keys)
        and bool(statuses)
        and len(keys) == len(statuses)
        and all(status in RESETTABLE_ROW_STATUSES for status in statuses)
    )


def add_reset_failed_action(
    menu: QMenu,
    *,
    keys: Sequence[object],
    statuses: Sequence[str],
    emit_reset: Callable[[list[object]], None],
):
    """Append the shared "Reset Failed Jobs..." action to ``menu``."""

    action = menu.addAction(RESET_FAILED_ACTION_TEXT)
    action.setEnabled(selection_supports_reset(keys, statuses))
    action.triggered.connect(
        lambda _checked=False, selected=list(keys): emit_reset(list(selected))
    )
    return action


@dataclass(frozen=True)
class RowMenuAction:
    """One extra per-row context-menu action beyond the shared reset action.

    Added so a table can offer a second, workflow-specific row action (for
    example "Continue relaxation…" on the two relaxation tables) without every
    other :func:`open_reset_failed_context_menu`/:func:`bind_reset_failed_context_menu`
    caller having to change shape — the default ``additional_actions=()`` on
    both functions keeps the other five tables byte-identical.

    ``is_enabled``/``trigger`` both receive the same resolved key list
    :func:`add_reset_failed_action` uses: one entry per selected row that
    yielded a non-``None`` key from ``key_for_row``, in row order. An action
    that (like Continue relaxation) only ever makes sense for one row at a
    time should start its ``is_enabled`` with a ``len(keys) == 1`` check
    rather than assume the caller enforces single selection.
    """

    text: str
    is_enabled: Callable[[Sequence[object]], bool]
    trigger: Callable[[Sequence[object]], None]


def add_row_menu_action(menu: QMenu, *, keys: Sequence[object], action: RowMenuAction) -> QAction:
    """Append one :class:`RowMenuAction` to ``menu``, gated on ``keys``."""

    qt_action = menu.addAction(action.text)
    qt_action.setEnabled(action.is_enabled(keys))
    qt_action.triggered.connect(
        lambda _checked=False, selected=list(keys), trigger=action.trigger: trigger(selected)
    )
    return qt_action


def open_reset_failed_context_menu(
    table: QTableWidget,
    position,
    *,
    key_for_row: Callable[[int], object | None],
    status_for_row: Callable[[int], str],
    emit_reset: Callable[[list[object]], None],
    additional_actions: Sequence[RowMenuAction] = (),
) -> QMenu | None:
    """Open a row menu on the workflow row under ``position``.

    The reset action is the in-table recovery affordance for a failed/stopped/
    cancelled job whose row would otherwise be stuck: the dashboard Reset
    button is the only other way back to ``defined``, and it disappears once
    the job is dismissed. ``additional_actions`` appends further per-row
    actions after it (see :class:`RowMenuAction`); the default ``()`` keeps
    every existing caller's menu single-action.

    Right-clicking a row outside the current selection selects that row first,
    matching the convergence table this was extracted from, so the action always
    describes what the user is pointing at. A row whose ``key_for_row`` returns
    ``None`` (a half-built or unlabeled row) contributes no key, which disables
    the action for the whole selection rather than resetting only its keyed
    siblings — see :func:`selection_supports_reset`.

    Returns the menu (for tests) or ``None`` when the click missed every row.
    """

    row = table.rowAt(position.y())
    if row < 0:
        return None

    selected_rows = _selected_table_rows(table)
    if row not in selected_rows:
        table.selectRow(row)
        selected_rows = (row,)

    keys = [key for row_index in selected_rows if (key := key_for_row(row_index)) is not None]
    statuses = [status_for_row(row_index) for row_index in selected_rows]

    menu = QMenu(table)
    add_reset_failed_action(menu, keys=keys, statuses=statuses, emit_reset=emit_reset)
    for action in additional_actions:
        add_row_menu_action(menu, keys=keys, action=action)
    menu.exec(table.viewport().mapToGlobal(position))
    return menu


def bind_reset_failed_context_menu(
    table: QTableWidget,
    *,
    key_for_row: Callable[[int], object | None],
    status_for_row: Callable[[int], str],
    emit_reset: Callable[[list[object]], None],
    additional_actions: Sequence[RowMenuAction] = (),
) -> None:
    """Install :func:`open_reset_failed_context_menu` as ``table``'s row menu.

    Also stamps :data:`RESET_FAILED_TABLE_TOOLTIP` on the table so the menu is
    discoverable, unless the caller already set a tooltip of its own.
    ``additional_actions`` is forwarded unchanged; see :class:`RowMenuAction`.
    """

    if not table.toolTip():
        table.setToolTip(RESET_FAILED_TABLE_TOOLTIP)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.customContextMenuRequested.connect(
        lambda position: open_reset_failed_context_menu(
            table,
            position,
            key_for_row=key_for_row,
            status_for_row=status_for_row,
            emit_reset=emit_reset,
            additional_actions=additional_actions,
        )
    )


def _selected_table_rows(table: QTableWidget) -> tuple[int, ...]:
    selection_model = table.selectionModel()
    if selection_model is None:
        return ()
    return tuple(sorted(index.row() for index in selection_model.selectedRows()))


def workflow_row_is_locked(status: str) -> bool:
    return str(status).strip().casefold() not in {"", "ready", "defined", "not generated"}


def style_workflow_configure_button(
    button: QPushButton,
    *,
    workflow_label: str,
    configured: bool,
    locked: bool,
    noun: str = "row",
    used: bool = False,
) -> None:
    """Apply the shared Configure/Configured text and tooltip to ``button``.

    ``noun`` names the per-row entity in the tooltip (default ``"row"``; e.g.
    ``"UEP source job"``). ``used`` selects the ``"parameters used for"`` phrasing
    of the locked (inspect) tooltip; the default ``"parameters for"`` phrasing is
    preserved for callers that do not opt in.
    """

    button.setText("Configured..." if configured or locked else "Configure...")
    inspect_phrase = (
        f"Inspect the {workflow_label} parameters used for this {noun}."
        if used
        else f"Inspect the {workflow_label} parameters for this {noun}."
    )
    button.setToolTip(
        inspect_phrase
        if locked
        else (
            f"Edit custom {workflow_label} parameters for this {noun}."
            if configured
            else f"Configure {workflow_label} parameters for this {noun}."
        )
    )
