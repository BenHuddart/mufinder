"""Project home page: what a project has found and where the user left off.

``ProjectHomeWidget`` shows, top to bottom: the project's name and host line
with ``Open Journal``; where the user left off, with a button that opens that
stage; the muon sites pinned to the Muon Sites summary beside the cell survey;
one line of the later workflows' results; and the newest Journal entries.

What it shows comes from :mod:`mufinder.services.project_summary`,
:mod:`mufinder.services.muon_sites_summary`,
:mod:`mufinder.services.project_survey` and :mod:`mufinder.services.journal`;
this module only lays it out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cell_survey_widget import CellSurveyWidget
from mufinder.app.theme import (
    SPACING,
    card_margins,
    controls_scroll_area,
    make_empty_state_label,
    style_hint_text,
    style_section_title,
)
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    fill_workflow_table,
    fit_table_height_to_rows,
)
from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.services.journal import short_day_label, recent_entries
from mufinder.services.muon_sites_summary import PinnedMuonSite, all_pinned_muon_sites
from mufinder.services.project_document import active_host_structure
from mufinder.services.project_summary import ProjectSummary, summarize_project
from mufinder.services.project_survey import discovered_sites_layer, pinned_sites_layer

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument

#: How many Journal entries ``Recent activity`` lists.
RECENT_ACTIVITY_ENTRIES = 5

#: The page that holds the pinned sites, which a site row opens.
MUON_SITES_ROUTE = "muon_sites"
JOURNAL_ROUTE = "provenance_log"

SITE_HEADERS = ("Site", "Charge state", "Wyckoff", "dE (eV)")
#: The site table always shows this many rows, and scrolls past the larger number.
MIN_SITE_ROWS = 3
MAX_SITE_ROWS = 10


def _card(parent: QWidget, caption: str) -> tuple[QFrame, QVBoxLayout]:
    """A card with a hint-text caption, and its layout."""

    card = QFrame(parent)
    card.setObjectName("overviewCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(card_margins())
    layout.setSpacing(SPACING.sm)
    label = QLabel(caption, card)
    style_hint_text(label)
    layout.addWidget(label)
    return card, layout


def _site_row(site: PinnedMuonSite) -> tuple[str, str, str, str]:
    energy = site.relative_energy_ev
    return (
        f"Site {site.rank}",
        muon_charge_state_label(site.charge_state).capitalize(),
        site.wyckoff_label or "—",
        "—" if energy is None else f"{energy:.3f}",
    )


class ProjectHomeWidget(QWidget):
    """The project home: header, where you left off, sites and survey, results, activity.

    ``set_document`` records the project; the page is recomputed when it is
    visible, so navigating elsewhere does not pay for it. Opening another
    project resets every section: no site stays selected and the survey shows
    the new host from the default view.
    """

    route_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectHome")
        self._document: ProjectDocument | None = None
        self._project_name = ""
        self._stale = True
        self._reset = True
        self._stage_key: str | None = None
        self._summary: ProjectSummary | None = None
        # A summary the caller already built for the document, used once.
        self._given_summary: ProjectSummary | None = None
        self._site_keys: list[tuple[str, int]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = controls_scroll_area(self)
        outer.addWidget(self._scroll)
        content = QWidget(self._scroll)
        content.setObjectName("projectHomeContent")
        self._scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)
        layout.addLayout(self._build_header(content))
        layout.addWidget(self._build_left_off(content))
        # The sites, the other results and the recent activity read down the left
        # column; the survey takes the right and the height they leave.
        columns = QHBoxLayout()
        columns.setSpacing(SPACING.md)
        reading = QVBoxLayout()
        reading.setSpacing(SPACING.md)
        reading.addWidget(self._build_sites(content))
        self.other_results_card, other_layout = _card(content, "Other results")
        self.other_results_label = QLabel(self.other_results_card)
        self.other_results_label.setObjectName("cardBody")
        self.other_results_label.setWordWrap(True)
        self.other_results_label.setTextFormat(Qt.TextFormat.PlainText)
        other_layout.addWidget(self.other_results_label)
        reading.addWidget(self.other_results_card)
        reading.addWidget(self._build_activity(content))
        reading.addStretch(1)
        columns.addLayout(reading, 1)
        self.survey = CellSurveyWidget(content)
        columns.addWidget(self.survey, 1)
        layout.addLayout(columns)
        layout.addStretch(1)

    # --- building -----------------------------------------------------------------

    def _build_header(self, parent: QWidget) -> QHBoxLayout:
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(SPACING.xs)
        self.name_label = QLabel(parent)
        style_section_title(self.name_label)
        self.name_label.setTextFormat(Qt.TextFormat.PlainText)
        titles.addWidget(self.name_label)
        self.host_label = QLabel(parent)
        style_hint_text(self.host_label)
        self.host_label.setWordWrap(True)
        titles.addWidget(self.host_label)
        header.addLayout(titles, 1)
        self.journal_button = QPushButton("Open Journal", parent)
        self.journal_button.setObjectName("projectHomeJournalButton")
        self.journal_button.setToolTip("Read the project's journal of actions and notes.")
        self.journal_button.clicked.connect(lambda: self.route_requested.emit(JOURNAL_ROUTE))
        header.addWidget(self.journal_button, 0, Qt.AlignmentFlag.AlignTop)
        return header

    def _build_left_off(self, parent: QWidget) -> QFrame:
        card, layout = _card(parent, "Where you left off")
        row = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(SPACING.xs)
        title_row = QHBoxLayout()
        self.stage_title = QLabel(card)
        self.stage_title.setObjectName("overviewCardTitle")
        title_row.addWidget(self.stage_title)
        self.stage_date = QLabel(card)
        style_hint_text(self.stage_date)
        title_row.addWidget(self.stage_date, 1, Qt.AlignmentFlag.AlignBottom)
        text.addLayout(title_row)
        self.stage_facts = QLabel(card)
        self.stage_facts.setObjectName("cardBody")
        self.stage_facts.setWordWrap(True)
        self.stage_facts.setTextFormat(Qt.TextFormat.PlainText)
        text.addWidget(self.stage_facts)
        row.addLayout(text, 1)
        self.open_stage_button = QPushButton(card)
        self.open_stage_button.setObjectName("projectHomeOpenStageButton")
        self.open_stage_button.clicked.connect(self._open_stage)
        row.addWidget(self.open_stage_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(row)
        return card

    def _build_sites(self, parent: QWidget) -> QFrame:
        card, layout = _card(parent, "Muon sites found")
        self.sites_table = create_workflow_table(
            SITE_HEADERS,
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
            accessible_name="Muon Sites Found",
            prefer_maximum_height=True,
        )
        self.sites_table.setParent(card)
        self.sites_table.setToolTip(
            "Select a site to find it in the survey; double-click it or press Return to "
            "open the Muon Sites summary."
        )
        self.sites_table.itemSelectionChanged.connect(self._highlight_selected_site)
        self.sites_table.doubleClicked.connect(lambda _index: self._open_sites())
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self.sites_table)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self._open_sites)
        layout.addWidget(self.sites_table)
        self.sites_progress = make_empty_state_label(
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, parent=card
        )
        self.sites_progress.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.sites_progress)
        return card

    def _build_activity(self, parent: QWidget) -> QFrame:
        self.activity_card, layout = _card(parent, "Recent activity")
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACING.lg)
        grid.setVerticalSpacing(SPACING.xs)
        grid.setColumnStretch(1, 1)
        self.activity_rows: list[tuple[QLabel, QLabel]] = []
        for row in range(RECENT_ACTIVITY_ENTRIES):
            date = QLabel(self.activity_card)
            style_hint_text(date)
            title = QLabel(self.activity_card)
            title.setObjectName("cardBody")
            title.setWordWrap(True)
            title.setTextFormat(Qt.TextFormat.PlainText)
            grid.addWidget(date, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(title, row, 1)
            self.activity_rows.append((date, title))
        layout.addLayout(grid)
        return self.activity_card

    # --- state ----------------------------------------------------------------------

    def set_document(
        self, document: ProjectDocument | None, summary: ProjectSummary | None = None
    ) -> None:
        """Show ``document``; ``summary``, when given, is its current summary.

        The page refreshes now if it is shown, else when it next is, from
        ``summary`` if one was given for the document as it is now.
        """

        self._given_summary = summary if document is not None else None
        if document is not self._document:
            self._reset = True
        self._document = document
        self._project_name = document.project.name if document is not None else ""
        self._stale = True
        if self.isVisible():
            self.refresh()

    def set_project_name(self, name: str) -> None:
        """Show ``name`` as the project's name, as after a rename."""

        self._project_name = name
        self.name_label.setText(name)

    def summary(self) -> ProjectSummary | None:
        return self._summary

    def refresh(self) -> None:
        self._stale = False
        document = self._document
        given, self._given_summary = self._given_summary, None
        if document is None:
            summary = None
        else:
            summary = given if given is not None else summarize_project(document)
        self._summary = summary
        reset, self._reset = self._reset, False
        if reset:
            self.sites_table.clearSelection()
            self._scroll.verticalScrollBar().setValue(0)
        self.name_label.setText(self._project_name)
        self.host_label.setText(summary.host_line if summary is not None else "")
        self._show_left_off(summary)
        self._show_sites(document, summary, keep_selection=not reset)
        other = summary.other_results if summary is not None else ""
        self.other_results_label.setText(other)
        self.other_results_card.setVisible(bool(other))
        self._show_activity(document)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        if self._stale:
            self.refresh()

    def _show_left_off(self, summary: ProjectSummary | None) -> None:
        stage = summary.last_stage if summary is not None else None
        self._stage_key = stage.key if stage is not None else None
        self.stage_title.setText(stage.title if stage is not None else "")
        self.stage_date.setText(
            short_day_label(stage.timestamp) if stage is not None and stage.timestamp else ""
        )
        self.stage_facts.setText(stage.fact_line if stage is not None else "")
        self.open_stage_button.setVisible(stage is not None)
        if stage is not None:
            self.open_stage_button.setText(f"Open {stage.title}")
            self.open_stage_button.setToolTip(f"Go to the {stage.title} page.")

    def _show_sites(
        self,
        document: ProjectDocument | None,
        summary: ProjectSummary | None,
        *,
        keep_selection: bool,
    ) -> None:
        sites = all_pinned_muon_sites(document)
        # The selection follows its site, not its row: pinning another analysis can
        # reorder the rows or drop the site.
        row = self._selected_row() if keep_selection else None
        kept = self._site_keys[row] if row is not None and row < len(self._site_keys) else None
        self._site_keys = [(site.charge_state, site.cluster_index) for site in sites]
        selected = self._site_keys.index(kept) if kept in self._site_keys else None
        self.sites_table.blockSignals(True)
        fill_workflow_table(self.sites_table, [_site_row(site) for site in sites])
        self.sites_table.clearSelection()
        self.sites_table.blockSignals(False)
        fit_table_height_to_rows(
            self.sites_table, minimum_rows=MIN_SITE_ROWS, maximum_rows=MAX_SITE_ROWS
        )
        self.sites_table.setVisible(bool(sites))
        self.sites_progress.setText("" if sites or summary is None else summary.site_progress)
        self.sites_progress.setVisible(not sites)
        if document is None:
            self.survey.set_survey(None, None)
        else:
            self.survey.set_survey(
                active_host_structure(document),
                pinned_sites_layer(document) if sites else discovered_sites_layer(document),
            )
        if selected is not None:
            self.sites_table.selectRow(selected)
        # Selecting a row highlights its site; with none selected, the survey clears it.
        self._highlight_selected_site()

    def _show_activity(self, document: ProjectDocument | None) -> None:
        entries = recent_entries(document, RECENT_ACTIVITY_ENTRIES) if document is not None else ()
        for index, (date, title) in enumerate(self.activity_rows):
            entry = entries[index] if index < len(entries) else None
            date.setText(short_day_label(entry.timestamp) if entry is not None else "")
            title.setText(entry.title if entry is not None else "")
            date.setVisible(entry is not None)
            title.setVisible(entry is not None)
        self.activity_card.setVisible(bool(entries))

    # --- actions --------------------------------------------------------------------

    def _selected_row(self) -> int | None:
        rows = self.sites_table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _highlight_selected_site(self) -> None:
        self.survey.highlight_marker(self._selected_row())

    def _open_stage(self) -> None:
        if self._stage_key is not None:
            self.route_requested.emit(self._stage_key)

    def _open_sites(self) -> None:
        if self.sites_table.rowCount():
            self.route_requested.emit(MUON_SITES_ROUTE)
