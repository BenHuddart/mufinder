"""The project navigator: the panel of workflow pages beside the page shown.

``ProjectNavigator`` is the panel: the project's name over a flat list. The
list starts with ``Project Home`` and ``Journal``, then each phase of the
workflow (``STAGE_PHASES``) as a heading over its stages, the stages of a group
one step in under the group's row, and each stage's saved workspaces and
comparison tables one step in under it. Every stage and group row carries a
seal for its status (not started, under way, complete), and a thin rail joins
the seals of a phase. The ``Host Structure`` row has a second, muted line
naming the active host as the Host Structure page's selector lists it, without
its date (``Relaxed — relax_1``); the full entry is in the row's tooltip and
accessible text. Without a host the row has one line. ``navigator_rows``
builds the rows from the stage catalogue in
:mod:`mufinder.services.project_summary`; the panel shows them
through ``_NavigatorModel`` and paints them with ``_NavigatorDelegate``.

The panel emits the route the user picks (``route_selected``, or
``home_requested`` for the ``Project Home`` row); the main window keeps the
route-to-page map and tells the panel which row to show as current, which
emits nothing. Workspace rows have a context menu that asks for their deletion
(``delete_analysis_requested``, ``delete_magnetic_workspace_requested``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractListModel,
    QItemSelectionModel,
    QModelIndex,
    QPointF,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QListView,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.eliding_label import ElidingLabel
from mufinder.app.icons import icon, workflow_stage_icon
from mufinder.app.theme import (
    NAVIGATOR_METRICS,
    SPACING,
    navigator_colours,
    navigator_detail_font,
    navigator_heading_font,
    navigator_row_font,
)
from mufinder.domain.routes import (
    comparison_route,
    magnetic_field_workspace_id_from_route,
    magnetic_field_workspace_route,
    relaxation_analysis_workspace_id_from_route,
    relaxation_analysis_workspace_route,
)
from mufinder.services.analysis_workspaces import analysis_workspaces
from mufinder.services.project_service import active_host_entry
from mufinder.services.project_summary import (
    STAGE_GROUPS,
    STAGE_PHASES,
    StageStatus,
    group_stage_keys,
    magnetic_field_workspaces,
    stage_group,
    stage_info,
    stage_statuses,
)

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument
    from mufinder.services.project_summary import ProjectSummary

#: The route of the ``Project Home`` row.
PROJECT_HOME_ROUTE = "project_home"
#: The route of the ``Journal`` row.
JOURNAL_ROUTE = "provenance_log"
#: The route of the ``Host Structure`` row, whose second line names the active host.
HOST_STRUCTURE_ROUTE = "host_structure"

#: How a row's status reads aloud, after its label: "Relaxations, under way".
_STATUS_WORDS = {
    StageStatus.NOT_STARTED: "not started",
    StageStatus.IN_PROGRESS: "under way",
    StageStatus.COMPLETE: "complete",
}

_GROUP_KEYS = frozenset(group.key for group in STAGE_GROUPS)


class RowKind(Enum):
    """What a navigator row is."""

    HEADING = "heading"
    SHELL = "shell"
    STAGE = "stage"
    WORKSPACE = "workspace"


@dataclass(frozen=True, slots=True)
class NavigatorRow:
    """One row of the navigator.

    ``route`` is the page the row opens, ``None`` for a phase heading.
    ``indent`` counts the steps the row sits in. ``status`` is a stage or
    group row's; ``optional`` marks a stage the main workflow does not pass
    through. ``tooltip`` is a stage's completion goal or a workspace's full
    name. ``rails`` lists the rail segments drawn through the row as
    ``(level, above, below)``: the indent level of the seals the rail joins,
    and whether it runs from the row's top and to its bottom. ``detail`` is a
    second, muted line under the label, and ``spoken_detail`` what a screen
    reader says for it after the status; both are empty for a one-line row.
    """

    kind: RowKind
    label: str
    route: str | None = None
    indent: int = 0
    status: StageStatus | None = None
    optional: bool = False
    tooltip: str = ""
    rails: tuple[tuple[int, bool, bool], ...] = ()
    detail: str = ""
    spoken_detail: str = ""

    @property
    def selectable(self) -> bool:
        return self.kind is not RowKind.HEADING

    @property
    def quiet(self) -> bool:
        """Whether the label is drawn muted: an optional stage not yet started."""

        return self.optional and self.status is StageStatus.NOT_STARTED

    @property
    def accessible_text(self) -> str:
        """The label with the row's status, as a screen reader says it."""

        parts = [self.label]
        if self.status is not None:
            parts.append(_STATUS_WORDS[self.status])
        if self.spoken_detail:
            parts.append(self.spoken_detail)
        return ", ".join(parts)


def navigator_rows(
    document: ProjectDocument | None, summary: ProjectSummary | None
) -> tuple[NavigatorRow, ...]:
    """Return the navigator's rows for ``document``, with the statuses of ``summary``.

    Without a summary every stage reads not started; without a document no
    stage has a workspace row, and without an active host the ``Host
    Structure`` row has no second line.
    """

    statuses = stage_statuses(summary) if summary is not None else {}
    host = active_host_entry(document) if document is not None else None
    rows = [
        NavigatorRow(RowKind.SHELL, "Project Home", PROJECT_HOME_ROUTE),
        NavigatorRow(RowKind.SHELL, "Journal", JOURNAL_ROUTE),
    ]

    def add_stage(key: str, indent: int) -> None:
        if key in _GROUP_KEYS:
            title, goal, optional = stage_group(key).title, stage_group(key).completion_goal, False
        else:
            info = stage_info(key)
            title, goal, optional = info.title, info.completion_goal, info.optional
        row = NavigatorRow(
            RowKind.STAGE,
            title,
            key,
            indent,
            statuses.get(key, StageStatus.NOT_STARTED),
            optional,
            f"Complete when: {goal}",
        )
        if key == HOST_STRUCTURE_ROUTE and host is not None:
            row = replace(
                row,
                detail=host.short_label or host.label,
                spoken_detail=f"active host {host.label}",
                tooltip=f"Active host: {host.label}\n{row.tooltip}",
            )
        rows.append(row)
        rows.extend(
            NavigatorRow(RowKind.WORKSPACE, label, route, indent + 1, tooltip=label)
            for route, label in _workspaces(document, key)
        )

    chains: list[list[int]] = []
    for phase in STAGE_PHASES:
        rows.append(NavigatorRow(RowKind.HEADING, phase.title))
        top_level: list[int] = []
        for member in phase.members:
            top_level.append(len(rows))
            add_stage(member, 0)
            if member in _GROUP_KEYS:
                children: list[int] = []
                for key in group_stage_keys(member):
                    children.append(len(rows))
                    add_stage(key, 1)
                chains.append(children)
        chains.append(top_level)
    return tuple(_with_rails(rows, chains))


def _workspaces(document: ProjectDocument | None, stage_key: str) -> list[tuple[str, str]]:
    """The ``(route, label)`` of each saved item shown under a stage, in the stage's order."""

    if document is None:
        return []
    if stage_key == "pristine_relaxation":
        tables = sorted(
            document.comparison_tables.values(),
            key=lambda table: (table.created_at, table.label.casefold()),
        )
        return [(comparison_route(table.id), table.label or "Comparison Table") for table in tables]
    if stage_key == "muon_analysis":
        return [
            (
                relaxation_analysis_workspace_route(str(workspace["id"]).strip()),
                str(workspace.get("name", "")).strip() or "Relaxation analysis",
            )
            for workspace in analysis_workspaces(document)
            if str(workspace.get("id", "")).strip()
        ]
    if stage_key == "magnetic_fields":
        return [
            (
                magnetic_field_workspace_route(str(workspace["id"]).strip()),
                str(workspace.get("title", "")).strip() or "Magnetic Fields",
            )
            for workspace in magnetic_field_workspaces(document)
            if str(workspace.get("id", "")).strip()
        ]
    return []


def _with_rails(rows: list[NavigatorRow], chains: list[list[int]]) -> list[NavigatorRow]:
    """Give each row the rail segments that join the seals of each chain through it."""

    rails: dict[int, list[tuple[int, bool, bool]]] = {}
    for chain in chains:
        if len(chain) < 2:
            continue
        level = rows[chain[0]].indent
        first, last = chain[0], chain[-1]
        for position in range(first, last + 1):
            rails.setdefault(position, []).append((level, position > first, position < last))
    return [
        replace(row, rails=tuple(sorted(rails[position]))) if position in rails else row
        for position, row in enumerate(rows)
    ]


class _NavigatorModel(QAbstractListModel):
    """The navigator's rows as a flat list model."""

    #: The item data role holding the row's ``NavigatorRow``.
    ROW_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: tuple[NavigatorRow, ...] = ()

    def rows(self) -> tuple[NavigatorRow, ...]:
        return self._rows

    def set_rows(self, rows: tuple[NavigatorRow, ...]) -> None:
        """Show ``rows``, updating the rows in place when the same routes are listed.

        A change of seal or label then repaints the rows without resetting the
        list, so its current row and scroll position stay as they are. A row
        that gains or loses its second line changes height, so the list lays
        its rows out again, still keeping its current row.
        """

        if rows == self._rows:
            return
        if [row.route for row in rows] == [row.route for row in self._rows]:
            relayout = [bool(row.detail) for row in rows] != [
                bool(row.detail) for row in self._rows
            ]
            if relayout:
                self.layoutAboutToBeChanged.emit()
            self._rows = rows
            if relayout:
                self.layoutChanged.emit()
            self.dataChanged.emit(self.index(0), self.index(len(rows) - 1))
            return
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def replace_row(self, position: int, row: NavigatorRow) -> None:
        rows = list(self._rows)
        rows[position] = row
        self._rows = tuple(rows)
        changed = self.index(position)
        self.dataChanged.emit(changed, changed)

    def position_of(self, route: str) -> int | None:
        return next(
            (position for position, row in enumerate(self._rows) if row.route == route), None
        )

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008 - Qt API.
        return 0 if parent.isValid() else len(self._rows)

    def flags(self, index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if not self._rows[index.row()].selectable:
            return Qt.ItemFlag.ItemNeverHasChildren
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemNeverHasChildren
        )

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row.label
        if role == Qt.ItemDataRole.ToolTipRole:
            return row.tooltip or None
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return row.accessible_text
        if role == Qt.ItemDataRole.UserRole:
            return row.route
        if role == self.ROW_ROLE:
            return row
        return None


def _row_icon(row: NavigatorRow):
    if row.route == PROJECT_HOME_ROUTE:
        return icon("project_home")
    if row.route == JOURNAL_ROUTE:
        return icon("journal")
    if row.status is StageStatus.COMPLETE:
        return workflow_stage_icon("completed")
    if row.status is StageStatus.IN_PROGRESS:
        return workflow_stage_icon("in_progress")
    return workflow_stage_icon("not_started")


class _NavigatorDelegate(QStyledItemDelegate):
    """Paint a row: its tint, rail, seal or icon and label, and size it.

    The selected row is tinted with an accent bar at its left edge and a bold
    label; the row under the pointer has a lighter tint. Seals keep their own
    colours in either. A row with a ``detail`` is taller by its second line,
    drawn muted in the caption size under the label; the seal and the rail's
    joint stay on the first line, and the tint and the bar cover both lines.
    Colours and sizes come from :mod:`mufinder.app.theme`.
    """

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802 - Qt API.
        row = index.data(_NavigatorModel.ROW_ROLE)
        if row is not None and row.kind is RowKind.HEADING:
            metrics = QFontMetrics(navigator_heading_font(option.font))
            return QSize(0, SPACING.md + metrics.height() + SPACING.xs)
        height = _first_line_height(option)
        if row is not None and row.detail:
            height += QFontMetrics(navigator_detail_font(option.font)).height()
        return QSize(0, height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        row = index.data(_NavigatorModel.ROW_ROLE)
        if row is None:
            return
        colours = navigator_colours()
        metrics = NAVIGATOR_METRICS
        rect = option.rect
        painter.save()
        try:
            if row.kind is RowKind.HEADING:
                font = navigator_heading_font(option.font)
                painter.setFont(font)
                painter.setPen(QColor(colours.heading_text))
                text_rect = rect.adjusted(SPACING.sm, SPACING.md, -SPACING.sm, -SPACING.xs)
                text = QFontMetrics(font).elidedText(
                    row.label, Qt.TextElideMode.ElideRight, text_rect.width()
                )
                painter.drawText(
                    text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text
                )
                return

            selected = bool(option.state & QStyle.StateFlag.State_Selected)
            hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
            if selected:
                painter.fillRect(rect, QColor(colours.selected_fill))
                bar = QRect(rect.left(), rect.top(), metrics.selection_bar, rect.height())
                painter.fillRect(bar, QColor(colours.selection_bar))
            elif hovered:
                painter.fillRect(rect, QColor(colours.hover_fill))

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(QPen(QColor(colours.rail), metrics.rail))
            # The seal, the rail's joint and the label sit on the first line.
            first_line = _first_line_height(option)
            centre_y = rect.top() + first_line / 2
            for level, above, below in row.rails:
                x = rect.left() + _seal_left(level) + metrics.seal // 2
                if above:
                    painter.drawLine(QPointF(x, rect.top()), QPointF(x, centre_y))
                if below:
                    painter.drawLine(QPointF(x, centre_y), QPointF(x, rect.bottom() + 1))

            if row.kind is not RowKind.WORKSPACE:
                seal = QRect(
                    rect.left() + _seal_left(row.indent),
                    rect.top() + (first_line - metrics.seal) // 2,
                    metrics.seal,
                    metrics.seal,
                )
                _row_icon(row).paint(painter, seal)

            font = navigator_row_font(option.font, selected=selected)
            painter.setFont(font)
            painter.setPen(
                QColor(colours.quiet_text if row.quiet and not selected else colours.text)
            )
            left = rect.left() + _label_left(row.indent)
            width = max(0, rect.right() - SPACING.sm - left)
            text_rect = QRect(left, rect.top(), width, first_line)
            text = QFontMetrics(font).elidedText(row.label, Qt.TextElideMode.ElideRight, width)
            painter.drawText(
                text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text
            )
            if row.detail:
                detail_font = navigator_detail_font(option.font)
                painter.setFont(detail_font)
                painter.setPen(QColor(colours.text if selected else colours.quiet_text))
                detail_metrics = QFontMetrics(detail_font)
                # Tucked under the label, into the first line's bottom padding.
                detail_rect = QRect(
                    left, rect.top() + first_line - SPACING.xs, width, detail_metrics.height()
                )
                painter.drawText(
                    detail_rect,
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                    detail_metrics.elidedText(row.detail, Qt.TextElideMode.ElideRight, width),
                )
        finally:
            painter.restore()


def _first_line_height(option: QStyleOptionViewItem) -> int:
    """The height of a row's first line: the whole height of a one-line row."""

    return max(NAVIGATOR_METRICS.seal, QFontMetrics(option.font).height()) + 2 * SPACING.xs


def _seal_left(indent: int) -> int:
    """The x of a seal at ``indent`` steps, from the row's left edge."""

    return SPACING.sm + indent * NAVIGATOR_METRICS.indent_step


def _label_left(indent: int) -> int:
    """The x of a label at ``indent`` steps; a workspace's lines up with its stage's children."""

    return _seal_left(indent) + NAVIGATOR_METRICS.seal + SPACING.sm


class _NavigatorListView(QListView):
    """The list, whose arrow, page and Home/End keys step over phase headings."""

    _BACKWARD = frozenset(
        {
            QAbstractItemView.CursorAction.MoveUp,
            QAbstractItemView.CursorAction.MovePageUp,
            QAbstractItemView.CursorAction.MovePrevious,
            QAbstractItemView.CursorAction.MoveLeft,
            QAbstractItemView.CursorAction.MoveEnd,
        }
    )

    def selectionCommand(self, index, event=None):  # noqa: N802 - Qt API naming.
        """Select the row that becomes current, whatever the modifier keys.

        A view with a single selection would otherwise let Ctrl move the
        current row without selecting it, or Ctrl+click clear the selection,
        and the highlighted row would no longer be the page shown.
        """

        if not index.isValid() or not index.flags() & Qt.ItemFlag.ItemIsSelectable:
            return QItemSelectionModel.SelectionFlag.NoUpdate
        return QItemSelectionModel.SelectionFlag.ClearAndSelect

    def moveCursor(self, action, modifiers):  # noqa: N802 - Qt API naming.
        index = super().moveCursor(action, modifiers)
        model = self.model()
        if model is None or not index.isValid():
            return index
        step = -1 if action in self._BACKWARD else 1
        position = index.row()
        while 0 <= position < model.rowCount():
            candidate = model.index(position, 0)
            if candidate.flags() & Qt.ItemFlag.ItemIsSelectable:
                return candidate
            position += step
        return self.currentIndex()


class ProjectNavigator(QFrame):
    """The navigator panel: the project's name over the list of pages.

    ``refresh`` rebuilds the rows from an open project and its summary.
    ``set_current_route`` shows a page's row as current without announcing it;
    the user's own choice of row is announced through ``route_selected`` or
    ``home_requested``.
    """

    #: The panel's width until the user drags it, and the narrowest it goes.
    DEFAULT_WIDTH = 200
    MINIMUM_WIDTH = 180

    route_selected = Signal(object)
    home_requested = Signal()
    delete_analysis_requested = Signal(str)
    delete_magnetic_workspace_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._has_document = False
        # True while the current row is set from code, which announces nothing.
        self._silent = False

        self._model = _NavigatorModel(self)
        self.view = _NavigatorListView()
        self.view.setObjectName("projectNavigatorList")
        self.view.setAccessibleName("Project navigator")
        self.view.setModel(self._model)
        self.view.setItemDelegate(_NavigatorDelegate(self.view))
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setMouseTracking(True)
        self.view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._open_context_menu)
        self.view.selectionModel().currentChanged.connect(self._on_current_changed)

        self.setObjectName("projectTreePanel")
        self.setMinimumWidth(self.MINIMUM_WIDTH)
        layout = QVBoxLayout(self)
        # The window gives the panel no bottom margin, so it would sit on the Job
        # Dashboard tray's header strip; this gap matches the pages' own.
        layout.setContentsMargins(0, 0, 0, SPACING.xs)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setObjectName("projectTreePanelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        self._title = ElidingLabel("", header)
        self._title.setObjectName("projectTreePanelTitle")
        header_layout.addWidget(self._title, 1)
        layout.addWidget(header)

        body = QFrame(self)
        body.setObjectName("projectTreePanelBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(1, 0, 1, 1)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.view, 1)
        layout.addWidget(body, 1)

        self.refresh(None, None)

    # --- header ---------------------------------------------------------------

    def set_title(self, name: str) -> None:
        """Show the project's ``name`` in the header, elided, with the full name as its tooltip."""

        self._title.setText(name)
        self._title.setToolTip(name)

    def title(self) -> str:
        return self._title.text()

    # --- rows -----------------------------------------------------------------

    def refresh(
        self,
        document: ProjectDocument | None,
        summary: ProjectSummary | None,
        *,
        keep_current: bool = True,
    ) -> bool:
        """Rebuild the rows for ``document`` and ``summary``; return whether the current row stays.

        With ``keep_current`` the current row stays current when its route is
        still listed; otherwise, or when it is gone, ``Project Home`` becomes
        current. Neither is announced. When the same routes are listed the
        rows are updated in place, so a refresh after a job update neither
        resets the list nor moves its scroll position.
        """

        previous = self.current_route() if keep_current else None
        self._has_document = document is not None
        self._silent = True
        try:
            self._model.set_rows(navigator_rows(document, summary))
        finally:
            self._silent = False
        kept = previous is not None and self._model.position_of(previous) is not None
        target = previous if kept else PROJECT_HOME_ROUTE
        if self.current_route() != target:
            self.set_current_route(target)
        return kept

    def rows(self) -> tuple[NavigatorRow, ...]:
        """Return every row, headings included, in the order shown."""

        return self._model.rows()

    def row(self, route: str) -> NavigatorRow | None:
        """Return the row that opens ``route``, or ``None`` when no row does."""

        position = self._model.position_of(route)
        return None if position is None else self._model.rows()[position]

    def has_route(self, route: str) -> bool:
        """Return whether a row opens ``route``."""

        return self._model.position_of(route) is not None

    def index_of(self, route: str) -> QModelIndex:
        """Return the list index of ``route``'s row; invalid when no row opens it."""

        position = self._model.position_of(route)
        return QModelIndex() if position is None else self._model.index(position)

    def set_row_label(self, route: str, label: str) -> None:
        """Relabel one route's row, leaving the current row and the page alone.

        A blank ``label`` leaves the row's label as it is.
        """

        position = self._model.position_of(route)
        if position is None:
            return
        row = self._model.rows()[position]
        label = label.strip() or row.label
        tooltip = label if row.kind is RowKind.WORKSPACE else row.tooltip
        self._model.replace_row(position, replace(row, label=label, tooltip=tooltip))

    # --- current row ----------------------------------------------------------

    def set_current_route(self, route: str) -> bool:
        """Make ``route``'s row current without announcing it.

        Returns ``False``, changing nothing, when no row opens ``route``.
        """

        index = self.index_of(route)
        if not index.isValid():
            return False
        self._silent = True
        try:
            self.view.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        finally:
            self._silent = False
        self.view.scrollTo(index)
        return True

    def current_route(self) -> str | None:
        """Return the current row's route, ``PROJECT_HOME_ROUTE`` on the project home."""

        route = self.view.currentIndex().data(Qt.ItemDataRole.UserRole)
        return route if isinstance(route, str) else None

    def active_route(self) -> str | None:
        """Return the route of the page the current row opens, ``None`` on the project home."""

        route = self.current_route()
        return None if route == PROJECT_HOME_ROUTE else route

    def focus_list(self) -> None:
        """Give the list keyboard focus."""

        self.view.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if self._silent or not current.isValid():
            return
        route = current.data(Qt.ItemDataRole.UserRole)
        if route == PROJECT_HOME_ROUTE:
            self.home_requested.emit()
        elif isinstance(route, str):
            self.route_selected.emit(route)

    # --- context menu ---------------------------------------------------------

    def _open_context_menu(self, position) -> QMenu | None:
        index = self.view.indexAt(position)
        route = index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None
        if not isinstance(route, str):
            return None
        menu = self.context_menu(route)
        if menu is not None:
            menu.exec(self.view.viewport().mapToGlobal(position))
        return menu

    def context_menu(self, route: str) -> QMenu | None:
        """Build the right-click menu for ``route``'s row, if it has one.

        Only user-created workspace rows (structural-relaxation analyses and
        saved magnetic field workspaces) can be deleted from the navigator;
        stages, comparison tables and the shell rows have no menu.
        """

        if not self._has_document or not self.has_route(route):
            return None
        menu = QMenu(self.view)
        if (analysis_id := relaxation_analysis_workspace_id_from_route(route)) is not None:
            action = menu.addAction("Delete Analysis…")
            action.triggered.connect(
                lambda _checked=False, wid=analysis_id: self.delete_analysis_requested.emit(wid)
            )
        elif (magnetic_id := magnetic_field_workspace_id_from_route(route)) is not None:
            action = menu.addAction("Delete Workspace…")
            action.triggered.connect(
                lambda _checked=False, wid=magnetic_id: (
                    self.delete_magnetic_workspace_requested.emit(wid)
                )
            )
        else:
            menu.deleteLater()
            return None
        return menu


__all__ = [
    "HOST_STRUCTURE_ROUTE",
    "JOURNAL_ROUTE",
    "PROJECT_HOME_ROUTE",
    "NavigatorRow",
    "ProjectNavigator",
    "RowKind",
    "navigator_rows",
]
