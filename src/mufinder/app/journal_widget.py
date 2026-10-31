"""Branded Journal view for project provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.eliding_label import ElidingLabel
from mufinder.app.theme import make_empty_state_label, style_hint_text, style_section_title
from mufinder.app.workflow_table_utils import create_workflow_table, fill_workflow_table
from mufinder.services.journal import (
    JournalDetails,
    JournalDocument,
    JournalEntry,
    build_journal,
    entry_details,
    entry_meta_text,
    export_journal,
    stamp_label,
)
from mufinder.domain.models import ProvenanceRecord

_COLLAPSED_CHEVRON = "▸"
_EXPANDED_CHEVRON = "▾"
#: Left of the title and the lines under it: the stamp dot and the gap after it.
_TEXT_INDENT = 18


class _JournalCard(QFrame):
    """One entry: a header line and a summary line, expanding to its details.

    Clicking the header, or Enter or Space while the card has focus, toggles it.
    The details are built on first expansion, so a page of collapsed cards costs
    only their two lines.
    """

    open_requested = Signal(str)
    expanded_changed = Signal(bool)

    def __init__(
        self,
        entry: JournalEntry,
        route_exists: Callable[[str], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entry = entry
        self._route_exists = route_exists
        self._expanded = False
        self._details: QWidget | None = None
        self._open_button: QPushButton | None = None

        self.setObjectName("journalEntryCard")
        self.setProperty("stampKind", entry.stamp_kind)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(f"{stamp_label(entry.stamp_kind)}: {entry.title}")
        self._sync_accessible_description()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)

        self._header = QWidget(self)
        self._header.setObjectName("journalCardHeader")
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        header = QHBoxLayout(self._header)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        stamp = QLabel(self._header)
        stamp.setObjectName("journalStamp")
        stamp.setProperty("stampKind", entry.stamp_kind)
        stamp.setToolTip(stamp_label(entry.stamp_kind))
        stamp.setAccessibleName(stamp_label(entry.stamp_kind))
        header.addWidget(stamp, 0, Qt.AlignmentFlag.AlignVCenter)

        self._title = ElidingLabel(entry.title, self._header)
        self._title.setObjectName("journalEntryTitle")
        header.addWidget(self._title, 1)

        meta = QLabel(entry_meta_text(entry), self._header)
        meta.setObjectName("journalWorkflowLabel")
        header.addWidget(meta, 0, Qt.AlignmentFlag.AlignVCenter)

        self._chevron = QLabel(_COLLAPSED_CHEVRON, self._header)
        self._chevron.setObjectName("journalChevron")
        header.addWidget(self._chevron, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._header)

        self._body = ElidingLabel(entry.card_summary, self)
        self._body.setObjectName("journalEntryBody")
        self._body.setContentsMargins(_TEXT_INDENT, 0, 0, 0)
        layout.addWidget(self._body)

    @property
    def entry(self) -> JournalEntry:
        return self._entry

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the details; the title and summary wrap while they show."""

        if expanded == self._expanded:
            return
        self._expanded = expanded
        if expanded and self._details is None:
            self._details = self._build_details(entry_details(self._entry))
            self.layout().addWidget(self._details)
        if self._details is not None:
            self._details.setVisible(expanded)
        if expanded:
            self.refresh_open_button()
        self._title.setWordWrap(expanded)
        self._body.setWordWrap(expanded)
        self._chevron.setText(_EXPANDED_CHEVRON if expanded else _COLLAPSED_CHEVRON)
        self._sync_accessible_description()
        self.expanded_changed.emit(expanded)

    def refresh_open_button(self) -> None:
        """Hide the link when the page it opens no longer exists."""

        if self._open_button is not None and self._entry.route is not None:
            self._open_button.setVisible(self._route_exists(self._entry.route))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton and self._header.geometry().contains(
            event.position().toPoint()
        ):
            self.set_expanded(not self._expanded)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        # A key a child ignored also arrives here, and it is not a toggle.
        focus = QApplication.focusWidget()
        child_has_focus = focus is not None and focus is not self and self.isAncestorOf(focus)
        if (
            not child_has_focus
            and not event.isAutoRepeat()
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
            and not event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        ):
            self.set_expanded(not self._expanded)
            return
        super().keyPressEvent(event)

    def _sync_accessible_description(self) -> None:
        state = "Expanded" if self._expanded else "Collapsed"
        self.setAccessibleDescription(f"{state}. {self._entry.card_summary}")

    def _build_details(self, details: JournalDetails) -> QWidget:
        block = QWidget(self)
        block.setObjectName("journalEntryDetails")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(_TEXT_INDENT, 6, 0, 2)
        layout.setSpacing(6)

        for text in details.lines:
            layout.addWidget(self._detail_label(text, "journalDetailLine", block))
        if details.job_table is not None:
            layout.addWidget(self._build_job_table(details, block))
        layout.addWidget(self._detail_label(details.exact_time, "journalDetailTime", block))

        if self._entry.route is not None and self._entry.open_label is not None:
            route = self._entry.route
            self._open_button = QPushButton(self._entry.open_label, block)
            self._open_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            # Outside a dialog a button ignores Return unless it is an auto default.
            self._open_button.setAutoDefault(True)
            self._open_button.clicked.connect(lambda: self.open_requested.emit(route))
            layout.addWidget(self._open_button)
        return block

    @staticmethod
    def _detail_label(text: str, object_name: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _build_job_table(self, details: JournalDetails, parent: QWidget) -> QTableWidget:
        job_table = details.job_table
        table = create_workflow_table(
            job_table.headers,
            resize_to_contents_columns=tuple(range(1, len(job_table.headers))),
            selection_mode=QAbstractItemView.SelectionMode.NoSelection,
            stretch_last_section=False,
            accessible_name="Calculations in this batch",
        )
        table.setParent(parent)
        fill_workflow_table(table, job_table.rows)
        for row in range(table.rowCount()):
            # The energy, force and warning columns are numbers.
            for column in range(2, table.columnCount()):
                table.item(row, column).setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The page scrolls, not the table: it is as tall as its rows.
        table.ensurePolished()
        table.resizeRowsToContents()
        height = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(height)
        # And it never clips a column: a narrow page scrolls sideways instead.
        table.setMinimumWidth(
            sum(
                max(
                    table.horizontalHeader().sectionSizeHint(column),
                    table.sizeHintForColumn(column),
                )
                for column in range(table.columnCount())
            )
            + 2 * table.frameWidth()
        )
        return table


class JournalWidget(QWidget):
    """Scrollable journal view derived from structured provenance.

    Each entry is a compact card that expands to its logged details and, when the
    entry's page exists (see :meth:`set_route_checker`), a link that emits
    ``open_route_requested``.

    The provenance stream is append-only, so the widget keeps the stream it last
    built from and its length: a refresh with the same, unchanged stream does
    nothing, and a grown stream touches only the cards it changes (a new record
    can add a card or update its batch's card). While the widget is hidden
    (another page of the main stack is shown), ``set_records`` only marks the view
    stale; it is brought up to date when it is shown.
    """

    open_route_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._route_checker: Callable[[str], bool] | None = None
        self._card_to_reveal: _JournalCard | None = None
        self._expanded_entries: set[tuple[datetime, str]] = set()
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setSingleShot(True)
        self._reveal_timer.setInterval(0)
        self._reveal_timer.timeout.connect(self._reveal_expanded_card)
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
        format_label.setBuddy(self._format_combo)
        controls.addWidget(self._format_combo)

        controls.addStretch(1)

        self._export_button = QPushButton("Export…", self)
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

    def set_route_checker(self, route_exists: Callable[[str], bool]) -> None:
        """Set what says whether a route's page exists; without it no card shows a link."""

        self._route_checker = route_exists

    def set_records(self, project_name: str, records: Sequence[ProvenanceRecord]) -> None:
        """Show the journal for ``records``, the project's append-only provenance stream.

        Passing the same sequence again after records were appended to it updates
        only the cards that changed; a different sequence (another project) rebuilds
        the view. A hidden widget defers the work until it is shown.
        """

        self._project_name = project_name
        self._records = records
        self._stale = True
        if records is not self._built_records:
            # Another stream rebuilds anyway; do not keep the old one alive meanwhile.
            self._built_records = None
            self._expanded_entries.clear()
        if not self.isHidden():
            self._refresh()

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API naming.
        # QStackedWidget shows its current page through setVisible, also while
        # the window itself is not shown yet, so the stale build runs here
        # rather than in showEvent.
        super().setVisible(visible)
        if visible and self._stale:
            self._refresh()
        if visible:
            # A table deleted while another page showed takes its link with it.
            for card in self._cards():
                card.refresh_open_button()

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
        journal = build_journal(self._project_name, records)
        updated = same_stream and self._update_cards(journal)
        self._journal = journal
        if not updated:
            self._rebuild_view()
        self._mark_built(records)

    def _mark_built(self, records: Sequence[ProvenanceRecord]) -> None:
        self._built_records = records
        self._built_count = len(records)
        self._built_last = records[-1] if records else None

    def _update_cards(self, journal: JournalDocument) -> bool:
        """Bring the cards up to ``journal``; ``False`` if that needs a rebuild.

        Updating works when the days and entries shown so far keep their places
        and anything new comes after them. A changed entry replaces its card.
        """

        days = self._journal.days
        if not days:
            return not journal.days
        if len(journal.days) < len(days):
            return False
        for index, day in enumerate(days):
            new_day = journal.days[index]
            if new_day.date_label != day.date_label or len(new_day.entries) < len(day.entries):
                return False
            if index < len(days) - 1 and len(new_day.entries) != len(day.entries):
                return False

        position = 0
        for day, new_day in zip(days, journal.days):
            position += 1  # The day header.
            for entry, new_entry in zip(day.entries, new_day.entries):
                if new_entry != entry:
                    self._replace_card(position, new_entry)
                position += 1
        # The trailing stretch stays last; new widgets go in front of it.
        for entry in journal.days[len(days) - 1].entries[len(days[-1].entries) :]:
            self._content_layout.insertWidget(position, self._build_entry_card(entry))
            position += 1
        for day in journal.days[len(days) :]:
            self._content_layout.insertWidget(position, self._build_day_header(day.date_label))
            position += 1
            for entry in day.entries:
                self._content_layout.insertWidget(position, self._build_entry_card(entry))
                position += 1
        return True

    def _replace_card(self, position: int, entry: JournalEntry) -> None:
        old_card = self._content_layout.takeAt(position).widget()
        focus = QApplication.focusWidget()
        had_focus = focus is old_card or (focus is not None and old_card.isAncestorOf(focus))
        card = self._build_entry_card(entry)
        if self._card_to_reveal is old_card:
            self._card_to_reveal = None
        old_card.hide()
        old_card.deleteLater()
        self._content_layout.insertWidget(position, card)
        if had_focus:
            # Focus only lands on a shown widget, and the layout shows a new one later.
            card.show()
            card.setFocus()

    def _rebuild_view(self) -> None:
        self._card_to_reveal = None
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

    def _build_entry_card(self, entry: JournalEntry) -> _JournalCard:
        card = _JournalCard(entry, self._route_exists, self._content)
        card.open_requested.connect(self.open_route_requested)
        # A card that was open comes back open, without pulling the page back to it.
        if self._entry_key(entry) in self._expanded_entries:
            card.set_expanded(True)
        card.expanded_changed.connect(self._on_card_expanded)
        return card

    @staticmethod
    def _entry_key(entry: JournalEntry) -> tuple[datetime, str]:
        """What stays the same when a card is rebuilt, as when its batch gains a completion."""

        return (entry.timestamp, entry.title)

    def _route_exists(self, route: str) -> bool:
        return self._route_checker is not None and self._route_checker(route)

    def _on_card_expanded(self, expanded: bool) -> None:
        """Bring a card that opened into view once the page has laid it out."""

        card = self.sender()
        if not isinstance(card, _JournalCard):
            return
        key = self._entry_key(card.entry)
        if expanded:
            self._expanded_entries.add(key)
            self._card_to_reveal = card
            self._reveal_timer.start()
        else:
            self._expanded_entries.discard(key)

    def _reveal_expanded_card(self) -> None:
        if self._card_to_reveal is not None:
            self._scroll.ensureWidgetVisible(self._card_to_reveal, 0, 0)
            self._card_to_reveal = None

    def _cards(self) -> list[_JournalCard]:
        return [
            item.widget()
            for index in range(self._content_layout.count())
            if isinstance((item := self._content_layout.itemAt(index)).widget(), _JournalCard)
        ]

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
