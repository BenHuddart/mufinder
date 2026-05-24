"""Tests for selection_snapshot.py — resolve_target_keys logic and widget snapshot."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem

from mufinder.app.selection_snapshot import (
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
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: called.append(1))

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


def test_resolve_prompt_yes_returns_all_eligible(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **kw: QMessageBox.StandardButton.Yes,
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


def test_resolve_prompt_no_returns_clicked_key_only(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **kw: QMessageBox.StandardButton.No,
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
        QMessageBox,
        "question",
        lambda _parent, _title, body, *a, **kw: prompts.append(body) or QMessageBox.StandardButton.Yes,
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
        QMessageBox,
        "question",
        lambda *a, **kw: QMessageBox.StandardButton.Yes,
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
        QMessageBox,
        "question",
        lambda *a, **kw: QMessageBox.StandardButton.Yes,
    )

    btn = _make_widget(qapp)
    btn.selection_snapshot_rows = (0, 1)

    # Build snapshot_keys from the rows as callers do
    snapshot_rows = btn.selection_snapshot_rows
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
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: called.append(1))
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
