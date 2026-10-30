"""Tests for the Journal widget."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QWidget,
)

from mufinder.app.journal_widget import JournalWidget
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord
from tests.support.journal import generation_record as _record


def _cards(widget: JournalWidget) -> list[QFrame]:
    return [
        card
        for card in widget.findChildren(QFrame, "journalEntryCard")
        if card.parentWidget() is widget._content
    ]


def _day_headers(widget: JournalWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel, "journalDayHeader")]


def _layout_order(widget: JournalWidget) -> list[str]:
    layout = widget._content_layout
    order = []
    for index in range(layout.count()):
        item_widget = layout.itemAt(index).widget()
        if item_widget is None:
            order.append("stretch")
        elif item_widget.objectName() == "journalDayHeader":
            order.append(item_widget.text())
        else:
            title = item_widget.findChild(QLabel, "journalEntryTitle")
            body = item_widget.findChild(QLabel, "journalEntryBody")
            order.append(f"card:{title.text() if title else ''}:{body.text() if body else ''}")
    return order


def test_journal_widget_shows_empty_state(qapp) -> None:
    widget = JournalWidget()
    widget.show()
    widget.set_records("EmptyProject", [])

    labels = [label.text() for label in widget.findChildren(type(widget._empty_label))]
    assert "No journal entries have been recorded for this project yet." in labels
    widget.close()


def test_journal_widget_survives_repeated_refreshes_after_deferred_deletes(qapp) -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="generate_candidates",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        parameters={"method": "random", "count": 2},
    )
    empty_text = "No journal entries have been recorded for this project yet."
    widget = JournalWidget()
    widget.show()

    for records in ([], [], [record], [], []):
        widget.set_records("RefreshProject", records)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        shown = [label.text() for label in widget.findChildren(QLabel) if not label.isHidden()]
        if records:
            assert "Muon sites generated" in shown
            assert empty_text not in shown
        else:
            assert empty_text in shown
            assert "Muon sites generated" not in shown

    widget.close()


def test_journal_widget_renders_day_headers_and_entries(qapp) -> None:
    widget = JournalWidget()
    widget.show()
    widget.set_records(
        "JournalProject",
        [
            ProvenanceRecord(
                timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
                action="generate_candidates",
                workflow_stage=WorkflowStage.MUON_GENERATION,
                parameters={"method": "random", "count": 2},
            )
        ],
    )

    texts = [label.text() for label in widget.findChildren(type(widget._empty_label))]
    assert "Tuesday 21 April 2026" in texts
    assert "Muon sites generated" in texts
    time_text = datetime(2026, 4, 21, 9, 0, tzinfo=UTC).astimezone().strftime("%H:%M")
    assert f"Muon Sites · {time_text}" in texts
    widget.close()


def test_journal_widget_exports_selected_format(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widget = JournalWidget()
    widget.set_records(
        "Export Project",
        [
            ProvenanceRecord(
                timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
                action="submit_jobs",
                workflow_stage=WorkflowStage.UEP,
                parameters={
                    "execution_target": "local",
                    "job_count": 1,
                    "job_labels": ["uep_row_1"],
                },
            )
        ],
    )
    widget._format_combo.setCurrentIndex(1)
    output_path = tmp_path / "export.md"
    monkeypatch.setattr(
        "mufinder.app.journal_widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "Markdown Files (*.md)"),
    )
    notices: list[str] = []
    monkeypatch.setattr(
        "mufinder.app.journal_widget.QMessageBox.information",
        lambda *args, **kwargs: notices.append("info"),
    )

    widget._export_journal()

    exported = output_path.read_text(encoding="utf-8")
    assert "# Export Project Journal" in exported
    assert "Calculations launched" in exported
    assert notices == ["info"]
    widget.close()


def test_journal_widget_skips_the_rebuild_when_the_stream_has_not_grown(qapp) -> None:
    records = [_record(9), _record(10)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    cards = _cards(widget)

    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert _cards(widget) == cards
    widget.close()


def test_journal_widget_appends_one_card_for_a_new_record(qapp) -> None:
    records = [_record(9), _record(10)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    cards = _cards(widget)

    records.append(_record(11, count=5))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    after = _cards(widget)
    assert after[:2] == cards
    assert len(after) == 3
    assert _day_headers(widget) == ["Tuesday 21 April 2026"]
    # The appended view matches a fresh build of the same stream.
    fresh = JournalWidget()
    fresh.show()
    fresh.set_records("Stream", list(records))
    assert _layout_order(widget) == _layout_order(fresh)
    assert _layout_order(widget)[-1] == "stretch"
    fresh.close()
    widget.close()


def test_journal_widget_appends_a_day_header_for_a_record_on_a_new_day(qapp) -> None:
    records = [_record(9)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    first_card = _cards(widget)[0]

    records.extend([_record(12, day=22), _record(13, day=22)])
    widget.set_records("Stream", records)

    assert _cards(widget)[0] is first_card
    assert _day_headers(widget) == ["Tuesday 21 April 2026", "Wednesday 22 April 2026"]
    assert len(_cards(widget)) == 3
    widget.close()


def test_journal_widget_rebuilds_for_an_older_record_or_another_stream(qapp) -> None:
    records = [_record(9, day=22)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    first_card = _cards(widget)[0]

    # A record that sorts before the last entry cannot be appended at the end.
    records.append(_record(9, day=21))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert first_card not in _cards(widget)
    assert _day_headers(widget) == ["Tuesday 21 April 2026", "Wednesday 22 April 2026"]

    # Another project's stream of the same length is a different stream.
    cards = _cards(widget)
    widget.set_records("Other", [_record(8), _record(9)])
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert not set(_cards(widget)) & set(cards)
    assert _day_headers(widget) == ["Tuesday 21 April 2026"]
    widget.close()


def test_hidden_journal_page_builds_when_it_is_shown(qapp, monkeypatch) -> None:
    # The journal is one page of the main stack; while another page is shown,
    # set_records only marks it stale.
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    widget = JournalWidget()
    stack.addWidget(widget)
    stack.show()
    builds: list[str] = []
    original = JournalWidget._rebuild_view

    def _counting_rebuild(self: JournalWidget) -> None:
        builds.append(self._journal.project_name)
        original(self)

    monkeypatch.setattr(JournalWidget, "_rebuild_view", _counting_rebuild)
    records = [_record(9)]

    widget.set_records("Hidden", records)
    records.append(_record(10))
    widget.set_records("Hidden", records)

    assert builds == []
    assert _cards(widget) == []

    stack.setCurrentWidget(widget)

    assert builds == ["Hidden"]
    assert len(_cards(widget)) == 2
    stack.close()


def test_hidden_journal_exports_the_current_stream(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = JournalWidget()
    widget.set_records("Hidden Export", [_record(9)])
    output_path = tmp_path / "export.txt"
    monkeypatch.setattr(
        "mufinder.app.journal_widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "Text Files (*.txt)"),
    )
    monkeypatch.setattr(
        "mufinder.app.journal_widget.QMessageBox.information", lambda *args, **kwargs: None
    )

    widget._export_journal()

    assert "Hidden Export Journal" in output_path.read_text(encoding="utf-8")
    widget.close()


def test_journal_widget_ignores_new_records_the_journal_does_not_show(qapp, monkeypatch) -> None:
    unshown = ProvenanceRecord(timestamp=datetime(2026, 4, 21, 8, 0, tzinfo=UTC), action="x")
    records = [unshown]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    builds: list[str] = []
    original = JournalWidget._rebuild_view

    def _counting_rebuild(self: JournalWidget) -> None:
        builds.append("rebuild")
        original(self)

    monkeypatch.setattr(JournalWidget, "_rebuild_view", _counting_rebuild)

    records.append(ProvenanceRecord(timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC), action="y"))
    widget.set_records("Stream", records)

    assert builds == []
    widget.close()


def test_journal_widget_rebuilds_when_the_last_record_was_replaced(qapp) -> None:
    records = [_record(9), _record(10)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)

    records[-1] = dataclasses.replace(records[-1], parameters={"method": "near_element"})
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    bodies = [label.text() for label in widget.findChildren(QLabel, "journalEntryBody")]
    assert bodies[-1] == "Generated 0 muon sites with 'Attach muon to element'."
    widget.close()


def _job_submission(job_ids: list) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="submit_jobs",
        workflow_stage=WorkflowStage.UEP,
        parameters={
            "execution_target": "local",
            "job_count": len(job_ids),
            "jobs": [
                {"job_id": str(job_id), "label": f"uep_{n}"} for n, job_id in enumerate(job_ids)
            ],
        },
    )


def _job_outcome(action: str, job_id, *, hour: int, day: int = 21) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, day, hour, 0, tzinfo=UTC),
        action=action,
        workflow_stage=WorkflowStage.UEP,
        parameters={"job_label": "uep", "execution_target": "local"},
        job_id=job_id,
    )


def test_a_completion_updates_only_its_batch_card(qapp) -> None:
    job_ids = [uuid4(), uuid4()]
    records = [_record(8), _job_submission(job_ids), _record(10)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    first, batch_card, last = _cards(widget)

    records.append(_job_outcome("job_completed", job_ids[0], hour=11))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    after = _cards(widget)
    assert len(after) == 3
    assert first in after
    assert last in after
    assert batch_card not in after
    assert _layout_order(widget)[2].endswith(" · 1 of 2 completed")

    fresh = JournalWidget()
    fresh.show()
    fresh.set_records("Stream", list(records))
    assert _layout_order(widget) == _layout_order(fresh)
    fresh.close()
    widget.close()


def test_a_failure_and_a_finished_line_append_after_updating_the_batch_card(qapp) -> None:
    job_ids = [uuid4(), uuid4()]
    records = [_job_submission(job_ids)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)

    records.append(_job_outcome("job_failed", job_ids[0], hour=10))
    records.append(_job_outcome("job_completed", job_ids[1], hour=9, day=22))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert _day_headers(widget) == ["Tuesday 21 April 2026", "Wednesday 22 April 2026"]
    titles = [label.text() for label in widget.findChildren(QLabel, "journalEntryTitle")]
    assert sorted(titles) == [
        "Calculation failed",
        "Calculations launched",
        "Finished: 1 completed, 1 failed",
    ]

    fresh = JournalWidget()
    fresh.show()
    fresh.set_records("Stream", list(records))
    assert _layout_order(widget) == _layout_order(fresh)
    assert _layout_order(widget)[-1] == "stretch"
    fresh.close()
    widget.close()


def test_a_second_dismissal_folds_into_the_card_of_the_first(qapp) -> None:
    def dismissal(minute: int) -> ProvenanceRecord:
        return ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, minute, tzinfo=UTC),
            action="job_dismissed",
            workflow_stage=WorkflowStage.UEP,
            parameters={"job_label": f"uep_{minute}"},
            job_id=uuid4(),
        )

    records = [dismissal(1)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)

    records.append(dismissal(2))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    cards = _cards(widget)
    assert len(cards) == 1
    assert cards[0].findChild(QLabel, "journalEntryTitle").text() == "Calculations dismissed"
    widget.close()


def _batch_records() -> tuple[list[ProvenanceRecord], list]:
    job_ids = [uuid4(), uuid4()]
    completed = [
        dataclasses.replace(
            _job_outcome("job_completed", job_id, hour=10 + n),
            parameters={"job_label": f"uep_{n}", "result_total_energy": -100.0 - n},
        )
        for n, job_id in enumerate(job_ids)
    ]
    return [_job_submission(job_ids), *completed], job_ids


def _details(card: QFrame) -> QWidget | None:
    return card.findChild(QWidget, "journalEntryDetails")


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        ("generate_candidates", "milestone"),
        ("job_completed", "success"),
        ("job_stopped", "warning"),
        ("job_failed", "danger"),
    ],
)
def test_a_card_and_its_stamp_carry_the_stamp_kind(qapp, action: str, kind: str) -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action=action,
        parameters={"job_label": "uep"},
    )
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stamps", [record])

    card = _cards(widget)[0]
    assert card.property("stampKind") == kind
    assert card.findChild(QLabel, "journalStamp").property("stampKind") == kind
    widget.close()


def test_the_theme_keeps_red_for_failed_entries_only(qapp) -> None:
    from mufinder.app.theme import _build_stylesheet, dark_palette, light_palette

    for palette in (light_palette(), dark_palette()):
        stylesheet = _build_stylesheet(palette)
        for selector, color in (
            ('QLabel#journalStamp[stampKind="milestone"]', palette.text_muted),
            ('QLabel#journalStamp[stampKind="success"]', palette.seal_gold),
            ('QLabel#journalStamp[stampKind="warning"]', palette.warning),
            ('QLabel#journalStamp[stampKind="danger"]', palette.danger),
            ('QFrame#journalEntryCard[stampKind="milestone"]', palette.border),
            ("QFrame#journalEntryCard:focus", palette.text_primary),
        ):
            rule = stylesheet.split(selector, 1)[1].split("}", 1)[0]
            assert color in rule, selector


def test_a_collapsed_card_is_compact_even_with_a_long_summary(qapp) -> None:
    from mufinder.app.theme import apply_theme

    apply_theme(qapp, "light")
    long_text = "A symmetry failure with a very long explanation. " * 20
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="symmetry_unavailable",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        description=long_text,
    )
    widget = JournalWidget()
    widget.resize(700, 600)
    widget.show()
    widget.set_records("Compact", [record])
    QCoreApplication.processEvents()

    card = _cards(widget)[0]
    assert card.height() < 60
    assert widget._content.minimumSizeHint().width() < 700
    # The label keeps the whole text, so expanding shows it.
    body = card.findChild(QLabel, "journalEntryBody")
    assert body.text() == long_text

    card.set_expanded(True)
    QCoreApplication.processEvents()
    assert card.height() > 60
    assert body.wordWrap()
    widget.close()


def test_details_are_built_on_first_expansion_only(qapp) -> None:
    widget = JournalWidget()
    widget.show()
    widget.set_records("Lazy", [_record(9)])
    card = _cards(widget)[0]
    assert _details(card) is None

    card.set_expanded(True)
    details = _details(card)
    assert details is not None and not details.isHidden()

    card.set_expanded(False)
    assert details.isHidden()
    card.set_expanded(True)
    assert _details(card) is details
    widget.close()


def test_clicking_the_header_toggles_a_card_and_clicking_details_does_not(qapp) -> None:
    widget = JournalWidget()
    widget.resize(800, 600)
    widget.show()
    widget.set_records("Click", [_record(9)])
    QCoreApplication.processEvents()
    card = _cards(widget)[0]
    header = card.findChild(QWidget, "journalCardHeader")

    QTest.mouseClick(header, Qt.MouseButton.LeftButton, pos=QPoint(40, 5))
    assert card.is_expanded()

    QCoreApplication.processEvents()
    QTest.mouseClick(_details(card), Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert card.is_expanded()

    QTest.mouseClick(header, Qt.MouseButton.LeftButton, pos=QPoint(40, 5))
    assert not card.is_expanded()
    widget.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space])
def test_enter_and_space_toggle_a_focused_card(qapp, key: Qt.Key) -> None:
    widget = JournalWidget()
    widget.show()
    widget.set_records("Keys", [_record(9)])
    card = _cards(widget)[0]
    assert card.focusPolicy() == Qt.FocusPolicy.StrongFocus

    QTest.keyClick(card, key)
    assert card.is_expanded()
    QTest.keyClick(card, key)
    assert not card.is_expanded()
    widget.close()


def test_keys_do_not_toggle_with_a_modifier_on_repeat_or_from_a_child(qapp) -> None:
    widget = JournalWidget()
    widget.set_route_checker(lambda route: True)
    widget.show()
    widget.set_records("Keys", [_record(9)])
    QCoreApplication.processEvents()
    card = _cards(widget)[0]
    card.setFocus()

    QTest.keyClick(card, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert not card.is_expanded()
    # A held key arrives as repeats after the first press.
    repeat = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, "", True
    )
    QCoreApplication.sendEvent(card, repeat)
    assert not card.is_expanded()
    QTest.keyClick(card, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    assert card.is_expanded()
    QCoreApplication.processEvents()

    table_records, _job_ids = _batch_records()
    widget.set_records("Keys", table_records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    batch_card = _cards(widget)[0]
    batch_card.set_expanded(True)
    QCoreApplication.processEvents()
    table = batch_card.findChild(QTableWidget)
    table.setFocus()
    QCoreApplication.sendEvent(
        table, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    )
    assert batch_card.is_expanded()
    widget.close()


def test_return_on_open_in_opens_the_page_and_leaves_the_card_open(qapp) -> None:
    widget = JournalWidget()
    widget.set_route_checker(lambda route: True)
    widget.show()
    widget.set_records("Keys", [_record(9)])
    QCoreApplication.processEvents()
    card = _cards(widget)[0]
    card.set_expanded(True)
    QCoreApplication.processEvents()
    opened: list[str] = []
    widget.open_route_requested.connect(opened.append)

    QTest.keyClick(card.findChild(QPushButton), Qt.Key.Key_Return)

    assert opened == ["muon_candidates"]
    assert card.is_expanded()
    widget.close()


def test_the_title_stays_plain_text_when_a_card_expands(qapp) -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="create_comparison_table",
        parameters={"table_label": "<b>x</b>", "dft_structure_count": 1},
    )
    widget = JournalWidget()
    widget.show()
    widget.set_records("Plain", [record])
    card = _cards(widget)[0]

    card.set_expanded(True)

    assert card.findChild(QLabel, "journalEntryBody").textFormat() == Qt.TextFormat.PlainText
    assert card.findChild(QLabel, "journalEntryTitle").textFormat() == Qt.TextFormat.PlainText
    widget.close()


def test_a_narrow_page_scrolls_rather_than_clipping_the_job_table(qapp) -> None:
    records, _job_ids = _batch_records()
    widget = JournalWidget()
    widget.resize(800, 500)
    widget.show()
    widget.set_records("Narrow", records)
    card = _cards(widget)[0]
    card.set_expanded(True)
    QCoreApplication.processEvents()

    table = card.findChild(QTableWidget)
    columns = sum(table.columnWidth(column) for column in range(1, table.columnCount()))
    assert table.minimumWidth() > columns
    assert widget._content.minimumSizeHint().width() >= table.minimumWidth()
    widget.close()


def test_an_expanded_batch_card_shows_its_job_table(qapp) -> None:
    records, _job_ids = _batch_records()
    widget = JournalWidget()
    widget.show()
    widget.set_records("Batch", records)

    card = _cards(widget)[0]
    card.set_expanded(True)

    table = card.findChild(QTableWidget)
    assert table.accessibleName()
    assert [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())] == [
        "Calculation",
        "Status",
        "ΔE (meV)",
        "Max force (eV/Å)",
        "Warnings",
    ]
    assert [table.item(row, 2).text() for row in range(2)] == ["1000.0", "0.0"]
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    widget.close()


def test_a_card_stays_expanded_when_a_completion_replaces_it(qapp) -> None:
    job_ids = [uuid4(), uuid4()]
    records = [_job_submission(job_ids)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    _cards(widget)[0].set_expanded(True)

    records.append(_job_outcome("job_completed", job_ids[0], hour=11))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    card = _cards(widget)[0]
    assert card.is_expanded()
    assert card.findChild(QTableWidget).item(0, 1).text() == "Completed"
    # Coming back open does not pull the page back to the card.
    assert not widget._reveal_timer.isActive() or widget._card_to_reveal is None
    widget.close()


def test_expansion_follows_the_entry_through_a_rebuild_and_not_to_another_project(qapp) -> None:
    records = [_record(9, day=22), _record(10, day=22)]
    widget = JournalWidget()
    widget.show()
    widget.set_records("Stream", records)
    _cards(widget)[1].set_expanded(True)

    # An older record moves every card down and rebuilds the view.
    records.append(_record(8, day=21))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert [card.is_expanded() for card in _cards(widget)] == [False, False, True]

    widget.set_records("Other", [_record(10, day=22)])
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert [card.is_expanded() for card in _cards(widget)] == [False]
    widget.close()


def test_focus_inside_a_replaced_card_stays_on_the_card(qapp) -> None:
    job_ids = [uuid4(), uuid4()]
    records = [_job_submission(job_ids)]
    widget = JournalWidget()
    widget.show()
    widget.activateWindow()
    widget.set_records("Stream", records)
    QCoreApplication.processEvents()
    card = _cards(widget)[0]
    card.set_expanded(True)
    QCoreApplication.processEvents()
    table = card.findChild(QTableWidget)
    table.setFocus()
    QCoreApplication.processEvents()
    if QApplication.focusWidget() is not table:
        widget.close()
        pytest.skip("this platform gives the window no keyboard focus")

    records.append(_job_outcome("job_completed", job_ids[0], hour=11))
    widget.set_records("Stream", records)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()

    assert QApplication.focusWidget() is _cards(widget)[0]
    widget.close()


def test_expanding_the_last_card_scrolls_it_into_view(qapp) -> None:
    records = [_record(9), *(_record(hour, count=hour) for hour in range(10, 22))]
    records.extend(_batch_records()[0])
    widget = JournalWidget()
    widget.resize(800, 300)
    widget.show()
    widget.set_records("Reveal", sorted(records, key=lambda record: record.timestamp))
    QCoreApplication.processEvents()
    bar = widget._scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    last = _cards(widget)[-1]
    before = bar.value()

    last.set_expanded(True)
    for _ in range(5):
        QCoreApplication.processEvents()

    assert bar.value() > before
    widget.close()


def test_open_in_emits_the_route_and_hides_for_a_missing_page(qapp) -> None:
    pages = {"muon_candidates"}
    widget = JournalWidget()
    widget.set_route_checker(lambda route: route in pages)
    widget.show()
    widget.set_records("Open", [_record(9)])
    card = _cards(widget)[0]
    card.set_expanded(True)
    button = card.findChild(QPushButton)
    assert button.text() == "Open in Muon Sites"
    assert not button.isHidden()
    opened: list[str] = []
    widget.open_route_requested.connect(opened.append)

    button.click()
    assert opened == ["muon_candidates"]

    # The page goes while the Journal is hidden; the link is gone when it comes back.
    widget.hide()
    pages.clear()
    widget.show()
    assert button.isHidden()
    widget.close()


def test_open_in_is_hidden_without_a_route_checker_or_a_workflow(qapp) -> None:
    no_workflow = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="set_recommended_xc_functional",
        parameters={"recommended_xc_functional": "PBE"},
    )
    widget = JournalWidget()
    widget.show()
    widget.set_records("Open", [_record(8), no_workflow])
    with_workflow, without = _cards(widget)

    with_workflow.set_expanded(True)
    without.set_expanded(True)

    assert with_workflow.findChild(QPushButton).isHidden()
    assert without.findChild(QPushButton) is None
    widget.close()
