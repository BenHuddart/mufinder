"""Branded Journal view for project provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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

from mufinder.app.theme import make_empty_state_label, style_hint_text, style_section_title
from mufinder.services.journal import (
    JournalDocument,
    JournalEntry,
    build_journal,
    export_journal,
    extend_journal,
)
from mufinder.domain.models import ProvenanceRecord


class JournalWidget(QWidget):
    """Scrollable journal view derived from structured provenance.

    The provenance stream is append-only, so the widget keeps the stream it last
    built from and its length: a refresh with the same, unchanged stream does
    nothing, and a grown stream appends cards for the new records only. While the
    widget is hidden (another page of the main stack is shown), ``set_records``
    only marks the view stale; it is brought up to date when it is shown.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._journal = JournalDocument(project_name="")
        self._project_name = ""
        self._records: Sequence[ProvenanceRecord] = ()
        self._built_records: Sequence[ProvenanceRecord] | None = None
        self._built_count = 0
        self._built_last: ProvenanceRecord | None = None
        self._stale = False

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

        self._empty_label = make_empty_state_label(
            "No journal entries have been recorded for this project yet.",
            object_name="journalEmptyState",
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            parent=self,
        )
        self._empty_label.hide()

    def set_records(self, project_name: str, records: Sequence[ProvenanceRecord]) -> None:
        """Show the journal for ``records``, the project's append-only provenance stream.

        Passing the same sequence again after records were appended to it adds
        cards for the new records only; a different sequence (another project)
        rebuilds the view. A hidden widget defers the work until it is shown.
        """

        self._project_name = project_name
        self._records = records
        self._stale = True
        if records is not self._built_records:
            # Another stream rebuilds anyway; do not keep the old one alive meanwhile.
            self._built_records = None
        if not self.isHidden():
            self._refresh()

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API naming.
        # QStackedWidget shows its current page through setVisible, also while
        # the window itself is not shown yet, so the stale build runs here
        # rather than in showEvent.
        super().setVisible(visible)
        if visible and self._stale:
            self._refresh()

    def _refresh(self) -> None:
        self._stale = False
        records = self._records
        count = len(records)
        # The stream is append-only, so its identity and length fingerprint what
        # was built; the last record's identity also catches a replaced tail.
        same_stream = (
            records is self._built_records
            and count >= self._built_count
            and (not self._built_count or records[self._built_count - 1] is self._built_last)
        )
        if same_stream and count == self._built_count:
            if self._journal.project_name != self._project_name:
                self._journal = replace(self._journal, project_name=self._project_name)
            return
        if same_stream:
            current = replace(self._journal, project_name=self._project_name)
            extended = extend_journal(current, records[self._built_count :])
            if extended is current:
                # Only records the journal does not show were added.
                self._journal = current
                self._mark_built(records)
                return
            if (
                extended is not None
                and current.days
                and self._append_cards(self._journal, extended)
            ):
                self._journal = extended
                self._mark_built(records)
                return
        self._journal = build_journal(self._project_name, records)
        self._rebuild_view()
        self._mark_built(records)

    def _mark_built(self, records: Sequence[ProvenanceRecord]) -> None:
        self._built_records = records
        self._built_count = len(records)
        self._built_last = records[-1] if records else None

    def _append_cards(self, current: JournalDocument, extended: JournalDocument) -> bool:
        """Add the cards ``extended`` has beyond ``current``; ``False`` if it cannot append.

        Appending works when only the last day gained entries or new days follow
        it; anything else needs a rebuild.
        """

        days = current.days
        if any(
            len(extended.days[index].entries) != len(day.entries)
            for index, day in enumerate(days[:-1])
        ):
            return False
        # The trailing stretch stays last; new widgets go in front of it.
        position = self._content_layout.count() - 1
        last = len(days) - 1
        for entry in extended.days[last].entries[len(days[last].entries) :]:
            self._content_layout.insertWidget(position, self._build_entry_card(entry))
            position += 1
        for day in extended.days[len(days) :]:
            self._content_layout.insertWidget(position, self._build_day_header(day.date_label))
            position += 1
            for entry in day.entries:
                self._content_layout.insertWidget(position, self._build_entry_card(entry))
                position += 1
        return True

    def _rebuild_view(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is self._empty_label:
                # The empty-state label lives for the widget's lifetime; detach it only.
                widget.hide()
            elif widget is not None:
                widget.deleteLater()

        if not self._journal.days:
            self._content_layout.addWidget(self._empty_label)
            self._empty_label.show()
            self._content_layout.addStretch(1)
            return

        for day in self._journal.days:
            self._content_layout.addWidget(self._build_day_header(day.date_label))
            for entry in day.entries:
                self._content_layout.addWidget(self._build_entry_card(entry))

        self._content_layout.addStretch(1)

    def _build_day_header(self, date_label: str) -> QLabel:
        day_label = QLabel(date_label, self._content)
        day_label.setObjectName("journalDayHeader")
        return day_label

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
        if self._stale:
            self._refresh()
        if not self._journal.project_name:
            QMessageBox.information(
                self, "No journal", "Open a project before exporting the Journal."
            )
            return

        format_name = str(self._format_combo.currentData() or "text")
        extension = ".md" if format_name == "markdown" else ".txt"
        suggested_name = f"{self._safe_name(self._journal.project_name)}-journal{extension}"
        selected_filter = (
            "Markdown Files (*.md)" if format_name == "markdown" else "Text Files (*.txt)"
        )
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
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in project_name.strip()
        )
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
