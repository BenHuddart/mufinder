"""Tests for selection_snapshot.py — resolve_target_keys logic and widget snapshot."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from mufinder.app import selection_snapshot
from mufinder.app.selection_snapshot import (
    SelectionScope,
    SelectionSnapshotCheckBox,
    SelectionSnapshotComboBox,
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)


# ---------------------------------------------------------------------------
# resolve_target_keys unit tests
# ---------------------------------------------------------------------------


def _make_widget(qapp):
    """Minimal snapshot widget for resolver tests (no real table needed)."""
    btn = SelectionSnapshotPushButton()
    return btn


def test_resolve_returns_clicked_key_when_not_in_snapshot(qapp):
    btn = _make_widget(qapp)
    btn.selection_snapshot_rows = ()
    result = resolve_target_keys(
        btn,
        clicked_key="b",
        snapshot_keys=["a", "c"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("b",)


def test_resolve_returns_clicked_key_when_snapshot_is_empty(qapp):
    btn = _make_widget(qapp)
    btn.selection_snapshot_rows = ()
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=[],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("a",)


def test_resolve_no_prompt_when_only_one_eligible(qapp, monkeypatch):
    called = []
    monkeypatch.setattr(
        selection_snapshot, "ask_selection_scope", lambda *a, **kw: called.append(1)
    )

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("a",)
    assert called == []


def test_resolve_prompt_all_returns_all_eligible(qapp, monkeypatch):
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b", "c"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("a", "b", "c")


def test_resolve_prompt_this_row_returns_clicked_key_only(qapp, monkeypatch):
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.CLICKED_ONLY,
    )

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="b",
        snapshot_keys=["a", "b", "c"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("b",)


def test_resolve_partial_eligibility_prompts_with_subset(qapp, monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda _parent, _title, body, *a, **kw: prompts.append(body) or SelectionScope.ALL_SELECTED,
    )

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b", "c", "d"],
        eligibility=lambda k: k in ("a", "c"),
        parent=btn,
    )
    assert result == ("a", "c")
    assert "2 of 4" in prompts[0]


def test_resolve_clears_snapshot_after_consumption(qapp, monkeypatch):
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    btn = _make_widget(qapp)
    btn.selection_snapshot_rows = (0, 1)
    btn.selection_snapshot_key = "a"

    resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b"],
        eligibility=lambda k: True,
        parent=btn,
    )

    assert btn.selection_snapshot_rows == ()
    assert btn.selection_snapshot_key is None


def test_resolve_snapshot_rows_cleared_after_consumption(qapp, monkeypatch):
    """After resolve_target_keys runs, widget.selection_snapshot_rows is empty.

    Callers build snapshot_keys from selection_snapshot_rows right before
    calling resolve_target_keys.  Verifying the rows are cleared means that
    if the caller were to call again without a new press, they would derive
    an empty snapshot_keys and get the single-row fallback.
    """
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    btn = _make_widget(qapp)
    btn.selection_snapshot_rows = (0, 1)

    # Callers build snapshot_keys from the snapshot rows: (0, 1) -> a, b
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ("a", "b")
    # After consumption, rows are cleared
    assert btn.selection_snapshot_rows == ()

    # A second call, now building snapshot_keys from the cleared rows, would
    # produce an empty snapshot — single-row fallback, no prompt
    called = []
    monkeypatch.setattr(
        selection_snapshot, "ask_selection_scope", lambda *a, **kw: called.append(1)
    )
    snapshot_rows_2 = list(btn.selection_snapshot_rows)  # empty
    result2 = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=snapshot_rows_2,
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result2 == ("a",)
    assert called == []


def test_resolve_prompt_cancel_returns_no_keys(qapp, monkeypatch):
    monkeypatch.setattr(
        selection_snapshot,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.CANCEL,
    )

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="b",
        snapshot_keys=["a", "b", "c"],
        eligibility=lambda k: True,
        parent=btn,
    )
    assert result == ()


def test_resolve_prompt_wording_names_action_count_and_noun(qapp, monkeypatch):
    asked: list[tuple[str, str, str, str]] = []

    def _fake(_parent, title, body, all_label, one_label):
        asked.append((title, body, all_label, one_label))
        return SelectionScope.ALL_SELECTED

    monkeypatch.setattr(selection_snapshot, "ask_selection_scope", _fake)

    btn = _make_widget(qapp)
    resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b", "c"],
        eligibility=lambda k: True,
        parent=btn,
        action="Resubmit",
        noun=("job", "jobs"),
    )
    resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b", "c"],
        eligibility=lambda k: k != "c",
        parent=btn,
        action="Remove",
    )

    assert asked[0] == (
        "Apply to selected jobs?",
        "Resubmit all 3 selected jobs?",
        "All 3 selected jobs",
        "This job only",
    )
    assert asked[1] == (
        "Apply to selected rows?",
        "Remove 2 of 3 selected rows? 1 is not eligible and will be skipped.",
        "2 eligible rows",
        "This row only",
    )


@pytest.mark.parametrize(
    ("button_index", "expected"),
    [
        (0, SelectionScope.ALL_SELECTED),
        (1, SelectionScope.CLICKED_ONLY),
        (2, SelectionScope.CANCEL),
    ],
)
def test_ask_selection_scope_maps_each_button(qapp, monkeypatch, button_index, expected):
    seen: dict[str, object] = {}

    def _fake_exec(box):
        buttons = box.buttons()
        seen["labels"] = [button.text() for button in buttons]
        seen["default"] = box.defaultButton().text()
        # QMessageBox.exec() records the clicked button; emulate a click.
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda _self: _ordered(box)[button_index])
        return 0

    def _ordered(box):
        by_text = {button.text(): button for button in box.buttons()}
        cancel = next(
            b
            for b in box.buttons()
            if b not in (by_text["All 3 selected rows"], by_text["This row only"])
        )
        return [by_text["All 3 selected rows"], by_text["This row only"], cancel]

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec)

    parent = QPushButton()
    result = selection_snapshot.ask_selection_scope(
        parent,
        "Apply to selected rows?",
        "Remove all 3 selected rows?",
        "All 3 selected rows",
        "This row only",
    )

    assert result is expected
    assert len(seen["labels"]) == 3
    assert seen["default"] == "All 3 selected rows"


# ---------------------------------------------------------------------------
# Widget snapshot capture tests
# ---------------------------------------------------------------------------


def _make_table_with_rows(n_rows: int) -> QTableWidget:
    table = QTableWidget(n_rows, 1)
    for row in range(n_rows):
        table.setItem(row, 0, QTableWidgetItem(str(row)))
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    return table


def _select_rows(table: QTableWidget, rows: list[int]) -> None:
    model = table.selectionModel()
    assert model is not None
    model.clearSelection()
    for row in rows:
        model.select(
            table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def _make_mouse_event():
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        __import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_snapshot_push_button_captures_multi_row_selection(qapp):
    table = _make_table_with_rows(3)
    _select_rows(table, [0, 2])

    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(0, 0, btn)
    btn.mousePressEvent(_make_mouse_event())

    assert btn.selection_snapshot_rows == (0, 2)


def test_snapshot_tool_button_captures_multi_row_selection(qapp):
    table = _make_table_with_rows(3)
    _select_rows(table, [1, 2])

    btn = SelectionSnapshotToolButton(table)
    table.setCellWidget(1, 0, btn)
    btn.mousePressEvent(_make_mouse_event())

    assert btn.selection_snapshot_rows == (1, 2)


def test_snapshot_combo_captures_multi_row_selection(qapp):
    table = _make_table_with_rows(3)
    _select_rows(table, [0, 1, 2])

    combo = SelectionSnapshotComboBox(table)
    table.setCellWidget(0, 0, combo)
    combo.mousePressEvent(_make_mouse_event())

    assert combo.selection_snapshot_rows == (0, 1, 2)


def test_snapshot_empty_when_no_table_parent(qapp):
    btn = SelectionSnapshotPushButton()
    btn.mousePressEvent(_make_mouse_event())
    assert btn.selection_snapshot_rows == ()


def test_snapshot_push_button_single_selection(qapp):
    table = _make_table_with_rows(3)
    _select_rows(table, [1])

    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(1, 0, btn)
    btn.mousePressEvent(_make_mouse_event())

    assert btn.selection_snapshot_rows == (1,)


# ---------------------------------------------------------------------------
# Targeting rule: highlighted rows are what is affected
# ---------------------------------------------------------------------------


def _selected(table: QTableWidget) -> list[int]:
    return sorted(index.row() for index in table.selectionModel().selectedRows())


def test_press_on_unselected_row_moves_selection_to_that_row(qapp):
    table = _make_table_with_rows(3)
    table.setColumnCount(2)
    _select_rows(table, [0, 1])

    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(2, 1, btn)
    btn.mousePressEvent(_make_mouse_event())

    assert _selected(table) == [2]
    assert btn.selection_snapshot_rows == (2,)


def test_press_on_selected_row_keeps_multi_selection(qapp):
    table = _make_table_with_rows(3)
    table.setColumnCount(2)
    _select_rows(table, [0, 2])

    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(2, 1, btn)
    btn.mousePressEvent(_make_mouse_event())

    assert _selected(table) == [0, 2]
    assert btn.selection_snapshot_rows == (0, 2)


@pytest.mark.parametrize(
    ("scope", "expected_selection"),
    [
        (SelectionScope.ALL_SELECTED, [0, 1, 2]),
        (SelectionScope.CANCEL, [0, 1, 2]),
        (SelectionScope.CLICKED_ONLY, [1]),
    ],
)
def test_resolve_sets_selection_to_match_the_answer(qapp, monkeypatch, scope, expected_selection):
    monkeypatch.setattr(selection_snapshot, "ask_selection_scope", lambda *a, **kw: scope)

    table = _make_table_with_rows(3)
    table.setColumnCount(2)
    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(1, 1, btn)
    _select_rows(table, [0, 1, 2])
    btn.mousePressEvent(_make_mouse_event())
    # Emulate the platform collapsing the selection to an unrelated row
    # between the press and the clicked slot.
    _select_rows(table, [0])

    resolve_target_keys(
        btn,
        clicked_key=1,
        snapshot_keys=list(btn.selection_snapshot_rows),
        eligibility=lambda k: True,
        parent=table,
    )

    assert _selected(table) == expected_selection


def test_snapshot_combo_restores_pre_interaction_index_without_signals(qapp):
    table = _make_table_with_rows(2)
    combo = SelectionSnapshotComboBox(table)
    combo.addItems(["a", "b", "c"])
    table.setCellWidget(0, 0, combo)
    combo.setCurrentIndex(1)

    combo.mousePressEvent(_make_mouse_event())
    changes: list[int] = []
    combo.currentIndexChanged.connect(changes.append)
    combo.setCurrentIndex(2)
    assert changes == [2]

    combo.restore_pre_interaction_index()

    assert combo.currentIndex() == 1
    assert changes == [2]


def test_snapshot_combo_captures_on_wheel(qapp):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    table = _make_table_with_rows(3)
    _select_rows(table, [0, 1])
    combo = SelectionSnapshotComboBox(table)
    combo.addItems(["a", "b"])
    table.setCellWidget(0, 0, combo)

    event = QWheelEvent(
        QPointF(1, 1),
        QPointF(1, 1),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    combo.wheelEvent(event)

    assert combo.selection_snapshot_rows == (0, 1)
    assert combo.pre_interaction_index == 0


@pytest.mark.parametrize(
    "widget_class",
    [SelectionSnapshotPushButton, SelectionSnapshotToolButton, SelectionSnapshotComboBox],
)
def test_real_click_on_cell_widget_keeps_multi_row_selection(qapp, widget_class):
    """A delivered mouse click must not collapse the selection before the snapshot.

    Regression: when a cell widget gains focus the view moves its current
    index to that cell, before the press arrives and again when a combo popup
    closes, so the selection was already down to the clicked row when it was
    captured.
    """
    from PySide6.QtTest import QTest

    table = _make_table_with_rows(3)
    table.setColumnCount(2)
    table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
    cell_widget = widget_class(table)
    if isinstance(cell_widget, SelectionSnapshotComboBox):
        cell_widget.addItems(["a", "b"])
    table.setCellWidget(1, 1, cell_widget)
    table.show()
    table.activateWindow()
    table.setFocus()
    qapp.processEvents()
    _select_rows(table, [0, 1, 2])

    # Focus arrives ahead of the press on a real click; deliver it explicitly so
    # the test does not depend on the platform plugin's window activation.
    cell_widget.setFocus(Qt.FocusReason.MouseFocusReason)
    QTest.mouseClick(cell_widget, Qt.MouseButton.LeftButton)
    if isinstance(cell_widget, SelectionSnapshotComboBox):
        cell_widget.hidePopup()

    assert _selected(table) == [0, 1, 2]
    assert cell_widget.selection_snapshot_rows == (0, 1, 2)
    table.close()


def test_focus_on_widget_in_unselected_row_lets_selection_move(qapp):
    from PySide6.QtTest import QTest

    table = _make_table_with_rows(3)
    table.setColumnCount(2)
    table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
    btn = SelectionSnapshotPushButton(table)
    table.setCellWidget(2, 1, btn)
    table.show()
    qapp.processEvents()
    _select_rows(table, [0, 1])

    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)

    assert _selected(table) == [2]
    table.close()


# ---------------------------------------------------------------------------
# confirm_unprompted: destructive actions are confirmed exactly once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("snapshot_keys", "answer", "expected"),
    [
        (["a"], True, ("a",)),  # single-row selection
        (["a"], False, ()),
        (["b", "c"], True, ("a",)),  # clicked row outside the selection
        (["b", "c"], False, ()),
    ],
)
def test_resolve_asks_confirm_unprompted_when_no_scope_prompt_is_shown(
    qapp, monkeypatch, snapshot_keys, answer, expected
):
    scope_prompts: list[int] = []
    monkeypatch.setattr(
        selection_snapshot, "ask_selection_scope", lambda *a, **kw: scope_prompts.append(1)
    )
    asked: list[int] = []

    btn = _make_widget(qapp)
    result = resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=snapshot_keys,
        eligibility=lambda k: True,
        parent=btn,
        confirm_unprompted=lambda: asked.append(1) or answer,
    )

    assert result == expected
    assert asked == [1]
    assert scope_prompts == []


@pytest.mark.parametrize("scope", list(SelectionScope))
def test_resolve_does_not_ask_confirm_unprompted_after_the_scope_prompt(qapp, monkeypatch, scope):
    monkeypatch.setattr(selection_snapshot, "ask_selection_scope", lambda *a, **kw: scope)
    asked: list[int] = []

    btn = _make_widget(qapp)
    resolve_target_keys(
        btn,
        clicked_key="a",
        snapshot_keys=["a", "b"],
        eligibility=lambda k: True,
        parent=btn,
        confirm_unprompted=lambda: asked.append(1) or True,
    )

    assert asked == []


def test_snapshot_check_box_captures_the_selection_and_its_state_on_press(qapp):
    table = QTableWidget(3, 2)
    for row in range(3):
        table.setItem(row, 0, QTableWidgetItem(str(row)))
    check = SelectionSnapshotCheckBox(table)
    table.setCellWidget(0, 1, check)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    selection = QItemSelection()
    for row in (0, 1):
        selection.select(table.model().index(row, 0), table.model().index(row, 0))
    table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    QTest.keyPress(check, Qt.Key.Key_Space)

    assert check.selection_snapshot_rows == (0, 1)
    assert check.pre_interaction_checked is False


def test_snapshot_check_box_restores_its_state_without_signals(qapp):
    check = SelectionSnapshotCheckBox()
    check.pre_interaction_checked = False
    check.setChecked(True)
    toggled: list[bool] = []
    check.toggled.connect(toggled.append)

    check.restore_pre_interaction_state()

    assert not check.isChecked()
    assert toggled == []


def test_snapshot_combo_ends_text_too_wide_for_it_in_an_ellipsis(qapp):
    combo = SelectionSnapshotComboBox()
    combo.addItem("YBa2Cu3O7 2x2x1 with a long name")
    combo.resize(90, 30)
    assert combo.displayed_text().endswith("…")

    combo.resize(600, 30)
    assert combo.displayed_text() == "YBa2Cu3O7 2x2x1 with a long name"
