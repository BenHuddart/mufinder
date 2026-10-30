"""Tests for the Journal widget."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QFrame, QLabel, QStackedWidget, QWidget

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
    assert "Muon Sites" in texts
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
