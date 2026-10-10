"""Shared helpers for workflow job/result tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import shiboken6
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from mufinder.app.theme import use_square_top_table_corners, use_tabbed_table_corners
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES

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
    AGENTS.md requires every delete/remove/cancel/reset shortcut to bind both
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


def partial_count_suffix(eligible: int, selected: int) -> str:
    """Return the " (2 of 5)" suffix a menu entry carries for a mixed selection.

    Empty when the action applies to every selected row or to none, so the
    count appears only when it tells the user something: that the entry will
    act on some of what is highlighted and skip the rest.
    """

    return f" ({eligible} of {selected})" if 0 < eligible < selected else ""


def resettable_keys(
    rows: Sequence[int],
    *,
    key_for_row: Callable[[int], object | None],
    status_for_row: Callable[[int], str],
) -> list[object]:
    """Return the submission keys of the selected rows a reset can act on.

    A row qualifies when its status is in :data:`RESETTABLE_ROW_STATUSES` and it
    has a key. A row without one (unlabeled or half-built, ``key_for_row``
    returned ``None``) has no job to reset and is skipped like any other row
    the action does not apply to.
    """

    keys: list[object] = []
    for row in rows:
        if status_for_row(row) not in RESETTABLE_ROW_STATUSES:
            continue
        key = key_for_row(row)
        if key is not None:
            keys.append(key)
    return keys


def add_reset_failed_action(
    menu: QMenu,
    *,
    keys: Sequence[object],
    selected_count: int,
    emit_reset: Callable[[list[object]], None],
):
    """Append the shared "Reset Failed Jobs..." action to ``menu``.

    ``keys`` are the resettable rows' keys (see :func:`resettable_keys`) and
    ``selected_count`` the size of the whole selection. The action is enabled
    when there is at least one key and resets those rows only; for a mixed
    selection its text carries the count, so the reset prompt that follows
    (which names the number of jobs) never covers more than the menu said.
    """

    action = menu.addAction(
        RESET_FAILED_ACTION_TEXT + partial_count_suffix(len(keys), selected_count)
    )
    action.setEnabled(bool(keys))
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
    #: Pass the selected table row indexes to ``is_enabled``/``trigger`` instead
    #: of submission keys. The standard actions built by
    #: :func:`standard_row_actions` use this, since configuring or removing a
    #: row has to work for a row that has no key yet.
    on_rows: bool = False
    #: Draw a separator above this action, to set table-specific actions apart
    #: from the shared core.
    separator_before: bool = False
    #: For an ``on_rows`` action: whether the action applies to a given row.
    #: When set, the action is offered for a mixed selection as long as one
    #: row qualifies, receives only the qualifying rows, and shows how many
    #: ("Cancel (2 of 5)"). ``is_enabled`` is then asked about those rows.
    applies_to_row: Callable[[int], bool] | None = None


#: Shared menu texts, so every job table names the same action identically.
RUN_LOCAL_ACTION_TEXT = "Run Local"
RUN_CLUSTER_ACTION_TEXT = "Run on Cluster"
CANCEL_ACTION_TEXT = "Cancel"
INSPECT_FAILURE_ACTION_TEXT = "Inspect Failure"
CONFIGURE_ACTION_TEXT = "Configure..."
REMOVE_ACTION_TEXT = "Remove"

_ACTIVE_ROW_STATUSES = frozenset({"submitting", "queued", "running", "retrieving"})
_INSPECTABLE_ROW_STATUSES = frozenset({"failed", "stopped"})


def standard_row_actions(
    *,
    status_for_row: Callable[[int], str],
    run_local: Callable[[Sequence[int]], None] | None = None,
    run_cluster: Callable[[Sequence[int]], None] | None = None,
    cancel: Callable[[Sequence[int]], None] | None = None,
    inspect: Callable[[Sequence[int]], None] | None = None,
    configure: Callable[[Sequence[int]], None] | None = None,
    remove: Callable[[Sequence[int]], None] | None = None,
    confirm_parent: QWidget | None = None,
) -> tuple[tuple[RowMenuAction, ...], tuple[RowMenuAction, ...]]:
    """Build the shared core of a job table's row menu.

    Returns ``(before_reset, after_reset)`` for
    :func:`bind_reset_failed_context_menu`, giving every table the same order:
    Run Local, Run on Cluster, Cancel, Inspect Failure, *Reset*, Configure...,
    Remove. A table omits an action by leaving its callback ``None``. Callbacks
    receive the selected table rows.

    Run, Cancel and Inspect are offered when at least one selected row is in a
    state the action applies to. They receive only those rows, and for a mixed
    selection the entry says how many ("Cancel (2 of 5)"). Configure and Remove
    apply to every selected row. Cancel asks for confirmation first (it needs
    ``confirm_parent``); Remove is expected to confirm in the table's own
    removal path, which already knows what each row has at stake.

    The run callbacks of the workflow tables act on the table selection rather
    than on the rows they are handed; that is equivalent, because the
    submission step drops rows that are not ready and reports how many.
    """

    def _status_is(predicate: Callable[[str], bool]) -> Callable[[int], bool]:
        return lambda row: predicate(status_for_row(row))

    def _confirmed_cancel(rows: Sequence[object]) -> None:
        phrase = count_phrase(len(rows), ("calculation", "calculations"))
        if confirm_parent is not None and not confirm_destructive(
            confirm_parent,
            "Cancel calculations",
            f"Cancel {phrase}?",
            "",
            f"Cancel {phrase}",
            "Keep running",
        ):
            return
        assert cancel is not None
        cancel([int(row) for row in rows])

    def _rows(callback: Callable[[Sequence[int]], None]) -> Callable[[Sequence[object]], None]:
        return lambda rows: callback([int(row) for row in rows])

    # Run follows the submit filter, not the edit lock: a cancelled row is
    # editable, but the dashboard's Resubmit/Reset is its way back to running.
    is_ready = _status_is(lambda status: not status or status in READY_WORKFLOW_ROW_STATUSES)
    before: list[RowMenuAction] = []
    if run_local is not None:
        before.append(
            RowMenuAction(
                RUN_LOCAL_ACTION_TEXT, bool, _rows(run_local), on_rows=True, applies_to_row=is_ready
            )
        )
    if run_cluster is not None:
        before.append(
            RowMenuAction(
                RUN_CLUSTER_ACTION_TEXT,
                bool,
                _rows(run_cluster),
                on_rows=True,
                applies_to_row=is_ready,
            )
        )
    if cancel is not None:
        before.append(
            RowMenuAction(
                CANCEL_ACTION_TEXT,
                bool,
                _confirmed_cancel,
                on_rows=True,
                applies_to_row=_status_is(lambda status: status in _ACTIVE_ROW_STATUSES),
            )
        )
    if inspect is not None:
        before.append(
            RowMenuAction(
                INSPECT_FAILURE_ACTION_TEXT,
                bool,
                _rows(inspect),
                on_rows=True,
                applies_to_row=_status_is(lambda status: status in _INSPECTABLE_ROW_STATUSES),
            )
        )

    after: list[RowMenuAction] = []
    if configure is not None:
        after.append(RowMenuAction(CONFIGURE_ACTION_TEXT, bool, _rows(configure), on_rows=True))
    if remove is not None:
        after.append(RowMenuAction(REMOVE_ACTION_TEXT, bool, _rows(remove), on_rows=True))
    return tuple(before), tuple(after)


def add_row_menu_action(
    menu: QMenu,
    *,
    keys: Sequence[object],
    action: RowMenuAction,
    rows: Sequence[int] = (),
) -> QAction:
    """Append one :class:`RowMenuAction` to ``menu``, gated on ``keys`` (or ``rows``)."""

    text = action.text
    if action.on_rows:
        keys = list(rows)
        if action.applies_to_row is not None:
            keys = [row for row in rows if action.applies_to_row(row)]
            text += partial_count_suffix(len(keys), len(rows))
    if action.separator_before:
        menu.addSeparator()
    qt_action = menu.addAction(text)
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
    emit_reset: Callable[[list[object]], None] | None,
    additional_actions: Sequence[RowMenuAction] = (),
    leading_actions: Sequence[RowMenuAction] = (),
) -> QMenu | None:
    """Open a row menu on the workflow row under ``position``.

    ``leading_actions`` are placed above the reset action and
    ``additional_actions`` below it, so a table using
    :func:`standard_row_actions` gets the shared order with reset in its place.
    A table with no failed-job reset of its own passes ``emit_reset=None`` and
    the reset action is left out.

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

    menu = QMenu(table)
    for action in leading_actions:
        add_row_menu_action(menu, keys=keys, action=action, rows=selected_rows)
    if emit_reset is not None:
        add_reset_failed_action(
            menu,
            keys=resettable_keys(
                selected_rows, key_for_row=key_for_row, status_for_row=status_for_row
            ),
            selected_count=len(selected_rows),
            emit_reset=emit_reset,
        )
    for action in additional_actions:
        add_row_menu_action(menu, keys=keys, action=action, rows=selected_rows)
    menu.exec(table.viewport().mapToGlobal(position))
    return menu


def bind_reset_failed_context_menu(
    table: QTableWidget,
    *,
    key_for_row: Callable[[int], object | None],
    status_for_row: Callable[[int], str],
    emit_reset: Callable[[list[object]], None] | None,
    additional_actions: Sequence[RowMenuAction] = (),
    leading_actions: Sequence[RowMenuAction] = (),
) -> None:
    """Install :func:`open_reset_failed_context_menu` as ``table``'s row menu.

    Also stamps :data:`RESET_FAILED_TABLE_TOOLTIP` on the table so the menu is
    discoverable, unless the caller already set a tooltip of its own.
    ``additional_actions`` is forwarded unchanged; see :class:`RowMenuAction`.
    """

    if not table.toolTip() and emit_reset is not None:
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
            leading_actions=leading_actions,
        )
    )


def _selected_table_rows(table: QTableWidget) -> tuple[int, ...]:
    selection_model = table.selectionModel()
    if selection_model is None:
        return ()
    return tuple(sorted(index.row() for index in selection_model.selectedRows()))


#: Dynamic property set on a row control that will act on the whole selection;
#: the theme draws it with an accent border.
MULTI_ROW_TARGET_PROPERTY = "multiRowTarget"

RUN_ALL_LOCAL_TEXT = "Run All Locally"
RUN_SELECTED_LOCAL_TEXT = "Run Selected Locally"
RUN_ALL_CLUSTER_TEXT = "Run All on Cluster"
RUN_SELECTED_CLUSTER_TEXT = "Run Selected on Cluster"
REMOVE_SELECTED_TEXT = "Remove Selected"


def selection_count_text(count: int) -> str:
    """Return the short count shown beside a table's buttons (empty for none)."""

    return f"{count} selected" if count > 0 else ""


class SelectionFeedback(QObject):
    """Keeps a job table's selection visible in the controls around it.

    Row text never changes with the selection (tables have to stay usable on a
    small screen), so everything that tells the user what a click will affect
    lives outside the rows and is kept current here:

    - ``count_label`` shows "N selected";
    - ``multi_row_hint`` is shown, with its text from ``multi_row_hint_text``,
      only while two or more rows are selected;
    - the run buttons read "Run All ..." with nothing selected and "Run
      Selected ..." otherwise, and are disabled only when the table is empty;
    - ``selection_buttons`` are enabled only while rows are selected;
    - row controls that will act on the whole selection carry
      :data:`MULTI_ROW_TARGET_PROPERTY` while two or more rows are selected.

    Call :meth:`refresh` after rebuilding a table's cell widgets, since new
    widgets start without the property. Once the table has been deleted (as
    when the app quits), :meth:`refresh` does nothing.

    Where several tables share one set of buttons (one table per tab), give
    each its own instance with ``is_active`` saying whether it is the table on
    show; only the active one drives the shared label and buttons, and
    :meth:`refresh` should be called on every instance when the tab changes.
    """

    def __init__(
        self,
        table: QTableWidget,
        *,
        count_label: QLabel | None = None,
        multi_row_hint: QLabel | None = None,
        multi_row_hint_text: Callable[[int], str] | None = None,
        run_local_button: QPushButton | None = None,
        run_cluster_button: QPushButton | None = None,
        selection_buttons: Sequence[QPushButton] = (),
        is_active: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(table)
        self._table = table
        self._is_active = is_active
        self._count_label = count_label
        self._multi_row_hint = multi_row_hint
        self._multi_row_hint_text = multi_row_hint_text
        self._run_local_button = run_local_button
        self._run_cluster_button = run_cluster_button
        self._selection_buttons = tuple(selection_buttons)

        selection_model = table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_table_changed)
        model = table.model()
        model.rowsInserted.connect(self._on_table_changed)
        model.rowsRemoved.connect(self._on_table_changed)
        model.modelReset.connect(self._on_table_changed)
        self.refresh()

    def _on_table_changed(self, *_args: object) -> None:
        self.refresh()

    def refresh(self) -> None:
        # The table's model is an older child than this object, so while the
        # table is being destroyed its model is torn down (emitting
        # ``modelReset``) before this object is disconnected.
        if not shiboken6.isValid(self._table):
            return
        rows = set(_selected_table_rows(self._table))
        count = len(rows)
        self._refresh_row_controls(rows)
        if self._is_active is not None and not self._is_active():
            return

        if self._count_label is not None:
            self._count_label.setText(selection_count_text(count))
        if self._multi_row_hint is not None:
            if count >= 2 and self._multi_row_hint_text is not None:
                self._multi_row_hint.setText(self._multi_row_hint_text(count))
            self._multi_row_hint.setVisible(count >= 2)

        has_rows = self._table.rowCount() > 0
        if self._run_local_button is not None:
            self._run_local_button.setText(RUN_SELECTED_LOCAL_TEXT if count else RUN_ALL_LOCAL_TEXT)
            self._run_local_button.setEnabled(has_rows)
        if self._run_cluster_button is not None:
            self._run_cluster_button.setText(
                RUN_SELECTED_CLUSTER_TEXT if count else RUN_ALL_CLUSTER_TEXT
            )
            self._run_cluster_button.setEnabled(has_rows)
        for button in self._selection_buttons:
            button.setEnabled(count > 0)

    def _refresh_row_controls(self, rows: set[int]) -> None:
        multi = len(rows) >= 2
        for row in range(self._table.rowCount()):
            for column in range(self._table.columnCount()):
                widget = self._table.cellWidget(row, column)
                if widget is None or not getattr(widget, "propagates_to_selection", False):
                    continue
                targeted = multi and row in rows
                if bool(widget.property(MULTI_ROW_TARGET_PROPERTY)) == targeted:
                    continue
                widget.setProperty(MULTI_ROW_TARGET_PROPERTY, targeted)
                # A dynamic property only restyles the widget once repolished.
                widget.style().unpolish(widget)
                widget.style().polish(widget)


def show_status_message(widget: QWidget, text: str, timeout_ms: int = 6000) -> None:
    """Show ``text`` in the status bar of the main window ``widget`` lives in.

    Does nothing for a widget that is not inside a main window (as in a test),
    and never creates a status bar a window does not already have.
    """

    window = widget.window()
    if not isinstance(window, QMainWindow):
        return
    status_bar = window.findChild(QStatusBar)
    if status_bar is not None:
        status_bar.showMessage(text, timeout_ms)


def multi_row_outcome_text(action: str, applied: int, selected: int) -> str:
    """Describe a finished multi-row action, e.g. for the status bar.

    ``action`` is a past-tense phrase ("Supercell set on"); ``selected`` is how
    many rows were selected and ``applied`` how many the action reached.
    """

    text = f"{action} {count_phrase(applied, ('row', 'rows'))}"
    skipped = selected - applied
    if skipped > 0:
        text += f", {count_phrase(skipped, ('row', 'rows'))} skipped"
    return text


# Row statuses whose parameters, supercell and sweep value stay editable: rows
# with no job yet, and cancelled rows, whose job never produced a result and may
# be edited before it runs again. Every other status locks the row.
_UNLOCKED_ROW_STATUSES = frozenset({"", "ready", "defined", "not generated", "cancelled"})


def workflow_row_is_locked(status: str) -> bool:
    """Return whether a workflow row with ``status`` is locked against edits.

    This is the one locked-row rule for every workflow table: a locked row's
    inline cells (supercell, sweep value) are read-only and its parameter dialog
    opens read-only. Whether a row can be submitted is a separate rule,
    ``READY_WORKFLOW_ROW_STATUSES`` in ``workflow_submission``.
    """

    return str(status).strip().casefold() not in _UNLOCKED_ROW_STATUSES


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
