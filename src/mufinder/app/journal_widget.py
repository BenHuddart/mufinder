"""Branded Journal view for project provenance."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_hint_text, style_section_title
from mufinder.services.journal import JournalDocument, JournalEntry, build_journal, export_journal
from mufinder.domain.models import ProvenanceRecord


class JournalWidget(QWidget):
    """Scrollable journal view derived from structured provenance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._journal = JournalDocument(project_name="")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Journal", self)
        style_section_title(title)
        layout.addWidget(title)

        self._subtitle = QLabel(
            "A day-by-day field notebook of the major calculations and workflow milestones recorded in this project.",
            self,
        )
        self._subtitle.setWordWrap(True)
        style_hint_text(self._subtitle)
        layout.addWidget(self._subtitle)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        format_label = QLabel("Export format", self)
        format_label.setObjectName("journalFormatLabel")
        controls.addWidget(format_label)

        self._format_combo = QComboBox(self)
        self._format_combo.addItem("Plain text (.txt)", "text")
        self._format_combo.addItem("Markdown (.md)", "markdown")
        controls.addWidget(self._format_combo)

        controls.addStretch(1)

        self._export_button = QPushButton("Export...", self)
        self._export_button.clicked.connect(self._export_journal)
        controls.addWidget(self._export_button)
        layout.addLayout(controls)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setObjectName("journalScrollArea")
        layout.addWidget(self._scroll, 1)

        self._content = QWidget(self._scroll)
        self._content.setObjectName("journalContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(12)
        self._scroll.setWidget(self._content)

        self._empty_label = QLabel(self)
        self._empty_label.setObjectName("journalEmptyState")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._empty_label.setText("No journal entries have been recorded for this project yet.")

    def set_records(self, project_name: str, records: list[ProvenanceRecord]) -> None:
        """Rebuild the visible journal from the supplied provenance stream."""

        self._journal = build_journal(project_name, records)
        self._rebuild_view()

    def _rebuild_view(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._journal.days:
            self._content_layout.addWidget(self._empty_label)
            self._content_layout.addStretch(1)
            return

        for day in self._journal.days:
            day_label = QLabel(day.date_label, self._content)
            day_label.setObjectName("journalDayHeader")
            self._content_layout.addWidget(day_label)

            for entry in day.entries:
                self._content_layout.addWidget(self._build_entry_card(entry))

        self._content_layout.addStretch(1)

    def _build_entry_card(self, entry: JournalEntry) -> QWidget:
        card = QFrame(self._content)
        card.setObjectName("journalEntryCard")
        card.setProperty("stampKind", entry.stamp_kind)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        stamp = QLabel(card)
        stamp.setObjectName("journalStamp")
        stamp.setProperty("stampKind", entry.stamp_kind)
        stamp.setText(self._stamp_text(entry.stamp_kind))
        top_row.addWidget(stamp, 0, Qt.AlignmentFlag.AlignLeft)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        title = QLabel(entry.title, card)
        title.setObjectName("journalEntryTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        if entry.workflow_label:
            workflow = QLabel(entry.workflow_label, card)
            workflow.setObjectName("journalWorkflowLabel")
            workflow.setWordWrap(True)
            layout.addWidget(workflow)

        body = QLabel(entry.body, card)
        body.setObjectName("journalEntryBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        return card

    def _export_journal(self) -> None:
        if not self._journal.project_name:
            QMessageBox.information(self, "No journal", "Open a project before exporting the Journal.")
            return

        format_name = str(self._format_combo.currentData() or "text")
        extension = ".md" if format_name == "markdown" else ".txt"
        suggested_name = f"{self._safe_name(self._journal.project_name)}-journal{extension}"
        selected_filter = "Markdown Files (*.md)" if format_name == "markdown" else "Text Files (*.txt)"
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Journal",
            suggested_name,
            "Text Files (*.txt);;Markdown Files (*.md)",
            selected_filter,
        )
        if not file_path:
            return

        try:
            export_journal(self._journal, Path(file_path), format_name=format_name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Saved Journal to:\n{file_path}")

    @staticmethod
    def _safe_name(project_name: str) -> str:
        cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in project_name.strip())
        return cleaned.strip("_") or "project"

    @staticmethod
    def _stamp_text(stamp_kind: str) -> str:
        labels = {
            "milestone": "Milestone",
            "success": "Completed",
            "warning": "Attention",
            "danger": "Failed",
        }
        return labels.get(stamp_kind, "Entry")
