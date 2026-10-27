"""Tests for the Journal widget."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QLabel

from mufinder.app.journal_widget import JournalWidget
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord


def test_journal_widget_shows_empty_state(qapp) -> None:
    widget = JournalWidget()
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
