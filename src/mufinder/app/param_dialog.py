"""Standalone CASTEP parameter dialog with progressive advanced controls."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import QModelIndex, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.help_dialog import ParameterHelpDialog
from mufinder.app.param_widgets import ParamEditorBase, create_param_editor
from mufinder.castep_io.gui_metadata import (
    gui_flag_is_explicitly_off,
    gui_flag_is_unset,
    gui_value_is_active,
    infer_task,
    keyword_category_label,
    keyword_locked_in_workflow,
    keyword_visible_for_task,
    keyword_visible_in_workflow,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.app.keyword_catalog_notice import CATALOG_SETUP_HINT
from mufinder.app.theme import track_item_view_corners
from mufinder.castep_io.schema import KeywordSpec, keyword_catalog_status

_ALWAYS_VISIBLE_ADVANCED_NAMES = ("spin_polarized",)
# Fallback preference order when the caller has not supplied a
# preferred_exclusive hint via set_values.  When both halves of an exclusive
# pair are pre-filled for UX toggle convenience, callers must pass the hint;
# this constant is consulted only when they do not (e.g. neither member has
# been configured yet and the dialog should default to a sensible choice).
_PREFERRED_EXCLUSIVE_NAMES = ("cut_off_energy", "kpoint_mp_grid")


@dataclass(slots=True)
class _DialogRow:
    spec: KeywordSpec
    editor: ParamEditorBase | None
    radio: QRadioButton | None
    locked: bool
    advanced: bool
    # The widgets the search filter shows or hides. A remove button the row
    # cannot use is left out so it stays hidden.
    cells: tuple[QWidget, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkflowTableRow:
    """A table row of a parameter dialog that holds no keyword.

    It sits directly below the row of the keyword ``after``, or last when that
    keyword has no row, and keeps its widgets when the rows are rebuilt. The
    search shows it while the query occurs in ``search_text``; without one it
    follows the row of ``after``. ``focus`` is the control the Tab key reaches
    in it, if any.
    """

    after: str
    label: QWidget
    value: QWidget
    search_text: str | None = None
    focus: QWidget | None = None


class _AdvancedSelectionDialog(QDialog):
    """Dialog that allows selecting one or more advanced parameters to add.

    Parameters are grouped under their category heading, both sorted by name,
    with the description wrapped in a second column.
    """

    def __init__(self, specs: list[KeywordSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Advanced Parameters")
        self.resize(640, 560)

        layout = QVBoxLayout(self)

        blurb = QLabel("Select one or more advanced parameters to add.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Search by name, keyword, description, or group")
        self._search.setAccessibleName("Search parameters")
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        self._tree = QTreeWidget(self)
        self._tree.setAccessibleName("Parameters to add")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Parameter", "Description"])
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tree.setRootIsDecorated(False)
        self._tree.setItemsExpandable(False)
        self._tree.setUniformRowHeights(False)
        self._tree.setWordWrap(True)
        self._tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._tree.setItemDelegate(_WrappedRowDelegate(self._tree))
        # Row heights follow the description column's width; a resize drag
        # re-wraps the rows once it settles, not on every pixel.
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.setInterval(0)
        self._relayout_timer.timeout.connect(self._tree.doItemsLayout)
        header.sectionResized.connect(self._on_section_resized)
        track_item_view_corners(self._tree)

        groups: dict[str, list[KeywordSpec]] = {}
        for spec in specs:
            groups.setdefault(keyword_category_label(spec.category), []).append(spec)
        heading_font = QFont(self._tree.font())
        heading_font.setBold(True)
        for heading in sorted(groups, key=str.casefold):
            group = QTreeWidgetItem([heading])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            group.setFont(0, heading_font)
            self._tree.addTopLevelItem(group)
            group.setFirstColumnSpanned(True)
            for spec in sorted(groups[heading], key=_spec_display_key):
                label = spec.label or spec.name.upper()
                item = QTreeWidgetItem([label, spec.description or ""])
                item.setData(0, Qt.ItemDataRole.UserRole, spec.name)
                item.setData(
                    0,
                    Qt.ItemDataRole.UserRole + 1,
                    " ".join(
                        [
                            spec.name,
                            *(spec.aliases or ()),
                            label,
                            spec.description or "",
                            heading,
                        ]
                    ).lower(),
                )
                item.setToolTip(0, spec.name.upper())
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemNeverHasChildren
                )
                item.setCheckState(0, Qt.CheckState.Unchecked)
                group.addChild(item)
        self._tree.expandAll()
        layout.addWidget(self._tree, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_section_resized(self, logical_index: int, old_size: int, new_size: int) -> None:
        if logical_index == 1 and old_size != new_size:
            self._relayout_timer.start()

    def _parameter_items(self) -> Iterator[QTreeWidgetItem]:
        for group_index in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(group_index)
            for child_index in range(group.childCount()):
                yield group.child(child_index)

    def selected_names(self) -> tuple[str, ...]:
        selected: list[str] = []
        for item in self._parameter_items():
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            name = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(name, str):
                selected.append(name)
        return tuple(selected)

    def _apply_filter(self) -> None:
        query = self._search.text().strip().lower()
        for group_index in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(group_index)
            any_visible = False
            for child_index in range(group.childCount()):
                item = group.child(child_index)
                haystack = str(item.data(0, Qt.ItemDataRole.UserRole + 1) or "")
                hidden = bool(query) and query not in haystack
                item.setHidden(hidden)
                any_visible = any_visible or not hidden
            group.setHidden(not any_visible)


class _WrappedRowDelegate(QStyledItemDelegate):
    """Size picker rows to the wrapped description and a whole check box.

    The default delegate sizes a row to one line of text, so a long description
    is cut short and the themed check box is clipped at the top and bottom.
    """

    def __init__(self, tree: QTreeWidget) -> None:
        super().__init__(tree)
        self._tree = tree

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        hint = super().sizeHint(option, index)
        style = self._tree.style()
        indicator_width = style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth, None, self._tree)
        indicator_height = style.pixelMetric(
            QStyle.PixelMetric.PM_IndicatorHeight, None, self._tree
        )
        width = hint.width()
        height = max(hint.height(), indicator_height + _ROW_PADDING)
        if index.column() == 0 and index.parent().isValid():
            # Measured: without this the themed check box pushes the longest
            # names into an ellipsis under ResizeToContents.
            width += indicator_width
        elif index.column() == 1:
            text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            available = max(self._tree.header().sectionSize(1) - 2 * _ROW_PADDING, 1)
            bounds = option.fontMetrics.boundingRect(
                QRect(0, 0, available, 1_000_000), int(Qt.TextFlag.TextWordWrap), text
            )
            height = max(height, bounds.height() + _ROW_PADDING)
        return QSize(width, height)


_ROW_PADDING = 6


def _spec_display_key(spec: KeywordSpec) -> tuple[str, str]:
    return ((spec.label or spec.name).casefold(), spec.name)


class ParameterDialog(QDialog):
    """Table-style parameter editor dialog with workflow constraints."""

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        workflow_stage: str,
        locked_values: dict[str, str] | None = None,
        intro_text: str | None = None,
        read_only: bool = False,
        editable_names: Iterable[str] | None = None,
        species_options: Sequence[str] = (),
        values: dict[str, Any] | None = None,
        preferred_exclusive: str | None = None,
        advanced_names: Iterable[str] | None = None,
        mixed_names: Iterable[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog and its rows once.

        ``values`` (with ``preferred_exclusive``), ``advanced_names`` and
        ``mixed_names`` are applied as :meth:`set_values`,
        :meth:`set_advanced_names` and :meth:`set_mixed_names` apply them, in
        that order, and the rows are then built once. Each exclusive group's
        choice therefore comes from the seed alone, never from the defaults of
        editors built along the way.
        """

        super().__init__(parent)
        # Rows are built once, at the end of __init__, after every seed is applied.
        self._rows_deferred = True
        self.setWindowTitle("CASTEP Parameters")
        self.resize(900, 720)

        self._workflow_stage = workflow_stage
        self._locked_values = dict(locked_values or {})
        locked_task = str(self._locked_values.get("task", "")).strip().upper()
        self._task = locked_task or infer_task(workflow_stage)
        self._read_only = bool(read_only)
        self._editable_names = (
            None
            if editable_names is None
            else {str(name).strip() for name in editable_names if str(name).strip()}
        )
        self._species_options = tuple(species_options)
        self._task_spec: KeywordSpec | None = None
        self._task_editor: ParamEditorBase | None = None

        self._specs_by_name: dict[str, KeywordSpec] = {}
        self._order: dict[str, int] = {}
        for index, spec in enumerate(specs):
            if spec.name == "task":
                self._task_spec = spec
                continue
            if spec.name in self._specs_by_name:
                continue
            self._specs_by_name[spec.name] = spec
            self._order[spec.name] = index

        self._blocked_names = self._compute_blocked_names()
        self._values: dict[str, Any] = {}
        self._added_advanced: set[str] = set()
        self._rows: dict[str, _DialogRow] = {}
        self._row_default_values: dict[str, object] = {}
        self._exclusive_groups: dict[tuple[str, ...], set[str]] = {}
        self._exclusive_selection: dict[tuple[str, ...], str] = {}
        # QButtonGroups handle Qt-level mutual exclusion across radios that
        # live in different parent widgets (each row's radio is parented to
        # its own label container, so Qt's auto-exclusivity does not span
        # rows by itself). Keyed by the same group_key used for selection.
        self._exclusive_button_groups: dict[tuple[str, ...], QButtonGroup] = {}
        # Name that should take priority in exclusive-group selection when
        # set by the caller via set_values(..., preferred_exclusive=...).
        self._preferred_exclusive: str | None = None
        # Parameters whose value differs between the rows being edited and that
        # the user has not touched yet (see set_mixed_names). Their "mixed"
        # marker labels are kept so they can be hidden as soon as one is edited.
        self._mixed_names: set[str] = set()
        self._mixed_markers: dict[str, QLabel] = {}

        root = QVBoxLayout(self)

        catalog_status = keyword_catalog_status()
        if catalog_status.is_starter:
            self._catalog_notice = QLabel(
                "Only MuFinder's limited built-in keyword set is available, so some CASTEP parameters are "
                f"not offered here. {CATALOG_SETUP_HINT}"
            )
            self._catalog_notice.setWordWrap(True)
            self._catalog_notice.setObjectName("catalogFallbackNotice")
            root.addWidget(self._catalog_notice)

        if intro_text:
            self._intro_label = QLabel(intro_text, self)
            self._intro_label.setWordWrap(True)
            self._intro_label.setObjectName("parameterDialogIntro")
            root.addWidget(self._intro_label)

        self._mixed_notice = QLabel(
            "Parameters marked 'mixed' differ between the selected rows. Each row keeps its "
            "own value for them unless you change the parameter here.",
            self,
        )
        self._mixed_notice.setWordWrap(True)
        self._mixed_notice.setObjectName("hintText")
        self._mixed_notice.hide()
        root.addWidget(self._mixed_notice)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task"))
        if self._task_spec is not None and not self._task_locked():
            self._task_editor = create_param_editor(
                self._task_spec,
                self,
                species_options=self._species_options,
            )
            self._task_editor.set_value(self._task)
            self._task_editor.value_changed.connect(self._on_task_changed)
            if self._read_only or not self._is_name_editable("task"):
                self._task_editor.setEnabled(False)
            task_row.addWidget(self._task_editor, 1)
        else:
            self._task_display = QLineEdit(self._task)
            self._task_display.setReadOnly(True)
            task_row.addWidget(self._task_display, 1)
        root.addLayout(task_row)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Filter by name, alias, description, or category")
        # The search only hides rows; every parameter stays in the dialog's
        # state, so hidden values are still collected and validated.
        self._search_input.textChanged.connect(lambda *_: self._apply_search_filter())
        search_row.addWidget(self._search_input, 1)
        root.addLayout(search_row)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._table_widget = QWidget(self._scroll_area)
        self._table_layout = QGridLayout(self._table_widget)
        self._table_layout.setContentsMargins(0, 0, 0, 0)
        self._table_layout.setColumnStretch(0, 2)
        self._table_layout.setColumnStretch(1, 4)
        self._table_layout.setColumnStretch(2, 0)
        self._table_layout.setColumnStretch(3, 0)
        self._table_layout.setColumnStretch(4, 0)
        self._scroll_area.setWidget(self._table_widget)
        root.addWidget(self._scroll_area, 1)
        self._workflow_rows = self._build_workflow_rows()

        controls_row = QHBoxLayout()
        self._add_advanced_button = QPushButton("+ Add Parameter", self)
        self._add_advanced_button.clicked.connect(self._open_add_advanced_dialog)
        controls_row.addWidget(self._add_advanced_button)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        if self._read_only:
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
            buttons.rejected.connect(self.reject)
        else:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=self,
            )
            buttons.accepted.connect(self._accept_if_valid)
            buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if values is not None:
            self.set_values(values, preferred_exclusive=preferred_exclusive)
        if advanced_names is not None:
            self.set_advanced_names(advanced_names)
        if mixed_names is not None:
            self.set_mixed_names(mixed_names)
        self._rows_deferred = False
        self._rebuild_rows(preserve_current=False)

    def _build_workflow_rows(self) -> tuple[WorkflowTableRow, ...]:
        """Return this dialog's rows that hold no keyword, parented to the table widget.

        Called once from the constructor, before any row is built.
        """

        return ()

    def set_values(
        self,
        values: dict[str, Any],
        *,
        preferred_exclusive: str | None = None,
    ) -> None:
        """Apply externally supplied values to known rows.

        ``preferred_exclusive`` names the member of an exclusive group that
        should be selected, overriding the built-in preference order.  Use
        this when the caller has pre-filled both halves of a mutually exclusive
        pair for UX toggling convenience and needs to preserve which half was
        the true active choice (e.g. ``"kpoint_mp_spacing"`` vs
        ``"kpoint_mp_grid"``).
        """

        for name, value in values.items():
            if name == "task":
                if self._task_editor is not None:
                    self._task_editor.set_value(value)
                    self._task = str(self._task_editor.value()).strip().upper()
                continue
            if name not in self._specs_by_name:
                continue
            self._values[name] = value
            if self._is_advanced_name(name):
                self._added_advanced.add(name)

        # Recompute exclusive defaults from newly provided values.
        self._preferred_exclusive = preferred_exclusive
        self._exclusive_selection.clear()
        self._rebuild_rows(preserve_current=False)

    def set_mixed_names(self, names: Iterable[str]) -> None:
        """Mark parameters whose value differs between the rows being edited.

        Call after :meth:`set_values` when one dialog edits several rows. A
        mixed parameter is shown with a "mixed" marker and is left out of
        :meth:`collect_values` until the user changes it, so the caller can keep
        each row's own value (see :meth:`untouched_mixed_names`). Editing it,
        resetting it, or switching the choice in its exclusive group clears the
        marker, and its value is then collected and applies to every row.
        """

        self._mixed_names = {str(name) for name in names if str(name) in self._specs_by_name}
        self._mixed_notice.setVisible(bool(self._mixed_names))
        self._rebuild_rows()

    def untouched_mixed_names(self) -> set[str]:
        """Return the mixed parameters the user left alone.

        Includes the other members of an exclusive group that has an untouched
        mixed member (for example ``kpoint_mp_spacing`` when ``kpoint_mp_grid``
        is mixed): the rows may differ in which member they use, so the group
        as a whole is left as it is on each row.
        """

        return {name for name in self._specs_by_name if self._is_untouched_mixed(name)}

    def _exclusive_partners(self, name: str) -> set[str]:
        """Return the parameters that cannot be set together with ``name``."""

        spec = self._specs_by_name.get(name)
        if spec is None:
            return set()
        partners: set[str] = set()
        for other_name, other in self._specs_by_name.items():
            if other_name == name:
                continue
            if (
                (spec.exclusive_group and other.exclusive_group == spec.exclusive_group)
                or other_name in spec.mutually_exclusive_with
                or name in other.mutually_exclusive_with
            ):
                partners.add(other_name)
        return partners

    def _is_untouched_mixed(self, name: str) -> bool:
        if not self._mixed_names:
            return False
        return name in self._mixed_names or bool(self._exclusive_partners(name) & self._mixed_names)

    def _mark_edited(self, name: str) -> None:
        """Record that the user set ``name``, so it now applies to every row."""

        if not self._is_untouched_mixed(name):
            return
        self._mixed_names -= {name, *self._exclusive_partners(name)}
        self._refresh_mixed_markers()

    def _refresh_mixed_markers(self) -> None:
        for name, marker in self._mixed_markers.items():
            marker.setVisible(self._is_untouched_mixed(name))

    def collect_values(self) -> dict[str, Any]:
        """Collect structured values for currently active rows.

        Rows hidden by the search filter are still collected. A mixed parameter
        the user did not touch is left out (see :meth:`set_mixed_names`). An
        unchecked flag is left out unless it is on by default; then it is
        collected as ``False``, so the workflow default does not turn it back on.
        """

        values: dict[str, Any] = {"task": self._task}
        for name, row in self._rows.items():
            if row.editor is None:
                continue
            if self._is_untouched_mixed(name):
                continue
            if self._in_inactive_exclusive_choice(name):
                continue
            value = row.editor.value()
            if value in (None, ""):
                continue
            if gui_flag_is_unset(row.spec, value):
                continue
            values[name] = value
        return values

    def validate(self) -> list[str]:
        """Validate field-level and cross-field constraints."""

        return validate_gui_values(self.collect_values(), self._specs_by_name, task=self._task)

    def set_advanced_names(self, names: Iterable[str]) -> None:
        """Set the advanced rows that should be visible in the dialog."""

        candidates = {
            name
            for name in names
            if name in self._specs_by_name
            and self._is_advanced_name(name)
            and name not in self._blocked_names
        }
        for name, value in self._values.items():
            if (
                name in self._specs_by_name
                and self._is_advanced_name(name)
                and name not in self._blocked_names
                and (
                    gui_value_is_active(self._specs_by_name[name], value)
                    # Only a row collects a value: without one, a flag stored
                    # as off would go back to its enabled default.
                    or gui_flag_is_explicitly_off(self._specs_by_name[name], value)
                )
            ):
                candidates.add(name)
        self._added_advanced = candidates
        self._rebuild_rows()

    def advanced_names(self) -> tuple[str, ...]:
        """Return currently visible, user-selected advanced row names."""

        return tuple(sorted(self._added_advanced, key=lambda name: self._order.get(name, 10_000)))

    def _accept_if_valid(self) -> None:
        errors = self.validate()
        if errors:
            QMessageBox.warning(self, "Invalid parameters", "\n".join(errors))
            return
        self.accept()

    def _is_basic_name(self, name: str) -> bool:
        spec = self._specs_by_name[name]
        if not keyword_visible_for_task(spec, task=self._task):
            return False
        return keyword_visible_in_workflow(
            spec,
            workflow_stage=self._workflow_stage,
            include_advanced=False,
        )

    def _is_advanced_name(self, name: str) -> bool:
        spec = self._specs_by_name[name]
        if not keyword_visible_for_task(spec, task=self._task):
            return False
        visible_basic = keyword_visible_in_workflow(
            spec,
            workflow_stage=self._workflow_stage,
            include_advanced=False,
        )
        visible_advanced = keyword_visible_in_workflow(
            spec,
            workflow_stage=self._workflow_stage,
            include_advanced=True,
        )
        return visible_advanced and not visible_basic

    def _is_locked_name(self, name: str) -> bool:
        if name in self._locked_values:
            return True
        spec = self._specs_by_name[name]
        return keyword_locked_in_workflow(spec, workflow_stage=self._workflow_stage)

    def _is_name_editable(self, name: str) -> bool:
        if self._read_only:
            return False
        if self._editable_names is None:
            return True
        return name in self._editable_names

    def _task_locked(self) -> bool:
        if self._task_spec is None:
            return True
        if "task" in self._locked_values:
            return True
        return keyword_locked_in_workflow(self._task_spec, workflow_stage=self._workflow_stage)

    def _compute_blocked_names(self) -> set[str]:
        locked_names = {
            name
            for name in self._specs_by_name
            if name in self._locked_values
            or keyword_locked_in_workflow(
                self._specs_by_name[name], workflow_stage=self._workflow_stage
            )
        }

        blocked = set()
        for name, spec in self._specs_by_name.items():
            if name in locked_names:
                continue
            if any(conflict in locked_names for conflict in spec.mutually_exclusive_with):
                blocked.add(name)
        return blocked

    def _visible_names(self) -> list[str]:
        """Return the parameters the dialog holds a row for, in display order.

        Independent of the search filter, which only hides rows (see
        :meth:`_displayed_names`).
        """

        names: list[str] = []
        for name in self._specs_by_name:
            if name in self._blocked_names:
                continue
            if self._is_basic_name(name):
                names.append(name)

        for name in self._specs_by_name:
            if name in self._blocked_names or name in names:
                continue
            if self._is_locked_name(name) and keyword_visible_in_workflow(
                self._specs_by_name[name],
                workflow_stage=self._workflow_stage,
                include_advanced=True,
            ):
                names.append(name)

        for name in self._added_advanced:
            if name in self._blocked_names or name in names:
                continue
            names.append(name)

        for name in _ALWAYS_VISIBLE_ADVANCED_NAMES:
            spec = self._specs_by_name.get(name)
            if (
                spec is None
                or name in self._blocked_names
                or name in names
                or not keyword_visible_for_task(spec, task=self._task)
                or not keyword_visible_in_workflow(
                    spec,
                    workflow_stage=self._workflow_stage,
                    include_advanced=True,
                )
            ):
                continue
            names.append(name)

        names.sort(key=lambda item: self._order.get(item, 10_000))
        return names

    def _row_matches_search(self, name: str) -> bool:
        query = self._search_input.text().strip().lower()
        return not query or self._matches_search(self._specs_by_name[name], query)

    def _displayed_names(self) -> list[str]:
        """Return the rows currently shown, i.e. those matching the search."""

        return [name for name in self._rows if self._row_matches_search(name)]

    def _apply_search_filter(self) -> None:
        # Hidden grid rows collapse to no height, so filtered rows leave no gap.
        for name, row in self._rows.items():
            shown = self._row_matches_search(name)
            for widget in row.cells:
                widget.setVisible(shown)
        query = self._search_input.text().strip().lower()
        for workflow_row in self._workflow_rows:
            if workflow_row.search_text is not None:
                shown = not query or query in workflow_row.search_text.lower()
            else:
                shown = workflow_row.after not in self._rows or self._row_matches_search(
                    workflow_row.after
                )
            workflow_row.label.setVisible(shown)
            workflow_row.value.setVisible(shown)

    def _group_key_for_name(self, name: str) -> tuple[str, ...] | None:
        for group_key, names in self._exclusive_groups.items():
            if name in names:
                return group_key
        return None

    def _in_inactive_exclusive_choice(self, name: str) -> bool:
        group_key = self._group_key_for_name(name)
        if group_key is None:
            return False
        selected = self._exclusive_selection.get(group_key)
        return selected is not None and selected != name

    def _detach_layout_item(self, item) -> None:
        widget = item.widget()
        if widget is not None:
            if any(widget in (row.label, row.value) for row in self._workflow_rows):
                return
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
            return

        nested_layout = item.layout()
        if nested_layout is None:
            return

        while nested_layout.count():
            nested_item = nested_layout.takeAt(0)
            self._detach_layout_item(nested_item)

    def _rebuild_rows(self, *, preserve_current: bool = True) -> None:
        if self._rows_deferred:
            return
        if preserve_current:
            self._capture_editor_values()

        previous_row_count = self._table_layout.rowCount()
        while self._table_layout.count():
            item = self._table_layout.takeAt(0)
            self._detach_layout_item(item)

        for row_index in range(previous_row_count + 2):
            self._table_layout.setRowStretch(row_index, 0)

        self._rows.clear()
        self._row_default_values.clear()
        self._mixed_markers.clear()
        # Detach and discard any prior button groups; new ones are created
        # below as rows are rebuilt. Reparenting to None lets Qt clean them
        # up alongside the rest of the row widgets.
        for button_group in self._exclusive_button_groups.values():
            button_group.setParent(None)
        self._exclusive_button_groups.clear()
        visible_names = self._visible_names()
        self._build_exclusive_groups(visible_names)

        parameter_header = QLabel("Parameter")
        parameter_header.setObjectName("parameterHeader")
        value_header = QLabel("Value")
        value_header.setObjectName("valueHeader")
        reset_header = QLabel("Reset")
        reset_header.setObjectName("resetHeader")
        info_header = QLabel("Info")
        info_header.setObjectName("infoHeader")

        self._table_layout.addWidget(parameter_header, 0, 0)
        self._table_layout.addWidget(value_header, 0, 1)
        self._table_layout.addWidget(reset_header, 0, 2)
        self._table_layout.addWidget(info_header, 0, 3)
        self._table_layout.addWidget(QLabel(""), 0, 4)

        row_indexes, workflow_row_indexes, row_count = self._table_row_indexes(visible_names)
        for name in visible_names:
            row_index = row_indexes[name]
            spec = self._specs_by_name[name]
            group_key = self._group_key_for_name(name)

            radio: QRadioButton | None = None
            label_text = spec.label or spec.name.upper()
            if group_key is not None:
                label_widget = QWidget(self._table_widget)
                label_layout = QHBoxLayout(label_widget)
                label_layout.setContentsMargins(0, 0, 0, 0)
                radio = QRadioButton(label_widget)
                radio.setAccessibleName(label_text)
                radio.setChecked(self._exclusive_selection.get(group_key) == name)
                radio.setEnabled(self._is_name_editable(name))
                radio.toggled.connect(
                    lambda checked, g=group_key, n=name: self._on_exclusive_toggled(g, n, checked)
                )
                # Each row's radio lives under its own label container; Qt
                # only auto-excludes siblings of the same parent, so we
                # explicitly register every radio in this group's
                # QButtonGroup to get cross-row mutual exclusion.
                button_group = self._exclusive_button_groups.get(group_key)
                if button_group is None:
                    button_group = QButtonGroup(self)
                    button_group.setExclusive(True)
                    self._exclusive_button_groups[group_key] = button_group
                button_group.addButton(radio)
                label_layout.addWidget(radio)
                label_layout.addWidget(QLabel(label_text, label_widget))
                label_layout.addStretch(1)
            else:
                label_widget = QLabel(label_text, self._table_widget)

            self._table_layout.addWidget(label_widget, row_index, 0)

            locked = self._is_locked_name(name)
            editor: ParamEditorBase | None = None
            if locked:
                raw_locked = self._locked_values.get(name, "Variable (set by workflow)")
                if isinstance(raw_locked, bool):
                    value_text = "TRUE" if raw_locked else "FALSE"
                else:
                    value_text = str(raw_locked)
                value_widget: QWidget = QLabel(value_text, self._table_widget)
                value_widget.setEnabled(False)
            else:
                editor = create_param_editor(
                    spec,
                    self._table_widget,
                    species_options=self._species_options,
                )
                self._row_default_values[name] = self._default_editor_value(spec)
                if name in self._values:
                    try:
                        editor.set_value(self._values[name])
                    except (TypeError, ValueError):
                        # Persisted/project-provided values can become stale if
                        # runtime metadata changes shape. Keep the dialog usable
                        # by leaving that editor at its current default.
                        pass
                editor.value_changed.connect(lambda param=name: self._on_row_edited(param))
                if self._in_inactive_exclusive_choice(name):
                    editor.setEnabled(False)
                elif not self._is_name_editable(name):
                    editor.setEnabled(False)
                value_widget = editor
                if self._mixed_names:
                    # The editor cannot show several values at once, so a
                    # marker beside it says the rows differ; it is hidden once
                    # the parameter is edited.
                    value_widget = QWidget(self._table_widget)
                    value_layout = QHBoxLayout(value_widget)
                    value_layout.setContentsMargins(0, 0, 0, 0)
                    value_layout.addWidget(editor, 1)
                    marker = QLabel("mixed", value_widget)
                    marker.setObjectName("mixedValueMarker")
                    marker.setToolTip(
                        "The selected rows have different values for this parameter. "
                        "Each row keeps its own value unless you change it here."
                    )
                    marker.setVisible(self._is_untouched_mixed(name))
                    value_layout.addWidget(marker)
                    self._mixed_markers[name] = marker

            self._table_layout.addWidget(value_widget, row_index, 1)

            reset_button = QPushButton("Reset", self._table_widget)
            reset_button.setMaximumWidth(58)
            reset_button.setEnabled(
                self._is_name_editable(name) and not locked and editor is not None
            )
            if self._is_name_editable(name) and not locked and editor is not None:
                reset_button.clicked.connect(
                    lambda _checked=False, param=name: self._reset_parameter(param)
                )
            self._table_layout.addWidget(reset_button, row_index, 2)

            info_button = QPushButton("Info", self._table_widget)
            info_button.setMaximumWidth(52)
            info_button.clicked.connect(
                lambda _checked=False, row_spec=spec: self._show_help(row_spec)
            )
            self._table_layout.addWidget(info_button, row_index, 3)

            remove_button = QToolButton(self._table_widget)
            remove_button.setText("x")
            remove_button.setAccessibleName(f"Remove {label_text}")
            can_remove = (
                self._is_name_editable(name)
                and self._is_advanced_name(name)
                and name in self._added_advanced
                and not locked
            )
            remove_button.setEnabled(can_remove)
            remove_button.setVisible(can_remove)
            if can_remove:
                remove_button.clicked.connect(
                    lambda _checked=False, param=name: self._remove_advanced(param)
                )
            self._table_layout.addWidget(remove_button, row_index, 4)

            self._rows[name] = _DialogRow(
                spec=spec,
                editor=editor,
                radio=radio,
                locked=locked,
                advanced=self._is_advanced_name(name),
                cells=(
                    label_widget,
                    value_widget,
                    reset_button,
                    info_button,
                    *((remove_button,) if can_remove else ()),
                ),
            )

        self._place_workflow_rows(workflow_row_indexes)
        self._table_layout.setRowStretch(row_count, 1)
        self._apply_search_filter()
        self._refresh_add_button_state()

    def _table_row_indexes(self, visible_names: list[str]) -> tuple[dict[str, int], list[int], int]:
        """Return the grid rows of the keywords and workflow rows, and the first free row."""

        below: dict[str, list[int]] = {}
        last: list[int] = []
        for position, workflow_row in enumerate(self._workflow_rows):
            if workflow_row.after in visible_names:
                below.setdefault(workflow_row.after, []).append(position)
            else:
                last.append(position)

        row_indexes: dict[str, int] = {}
        workflow_row_indexes = [0] * len(self._workflow_rows)
        next_index = 1
        for name in visible_names:
            row_indexes[name] = next_index
            next_index += 1
            for position in below.get(name, ()):
                workflow_row_indexes[position] = next_index
                next_index += 1
        for position in last:
            workflow_row_indexes[position] = next_index
            next_index += 1
        return row_indexes, workflow_row_indexes, next_index

    def _place_workflow_rows(self, row_indexes: list[int]) -> None:
        for workflow_row, row_index in zip(self._workflow_rows, row_indexes, strict=True):
            self._table_layout.addWidget(workflow_row.label, row_index, 0)
            self._table_layout.addWidget(workflow_row.value, row_index, 1, 1, 4)
            # The widgets outlive the rebuilt rows, so they would otherwise keep
            # their place at the start of the tab order.
            row = self._rows.get(workflow_row.after) or next(reversed(self._rows.values()), None)
            if row is not None and workflow_row.focus is not None:
                QWidget.setTabOrder(row.cells[-1], workflow_row.focus)

    def _show_help(self, spec: KeywordSpec) -> None:
        dialog = ParameterHelpDialog(spec, self)
        dialog.exec()

    def _default_editor_value(self, spec: KeywordSpec) -> object:
        if spec.default_value is not None:
            try:
                return normalize_gui_value(spec, spec.default_value)
            except (TypeError, ValueError):
                pass

        probe_editor = create_param_editor(spec, self, species_options=self._species_options)
        try:
            return probe_editor.value()
        finally:
            probe_editor.setParent(None)
            probe_editor.deleteLater()

    def _reset_parameter(self, name: str) -> None:
        row = self._rows.get(name)
        if row is None or row.editor is None:
            return

        default_value = self._row_default_values.get(name)
        if default_value is None:
            return

        try:
            row.editor.set_value(default_value)
        except (TypeError, ValueError):
            return

        self._values[name] = row.editor.value()
        # Resetting is a deliberate choice of the default for every row, even
        # when the editor already showed it and so emitted no change. The
        # inactive member of an either/or pair is the exception: its value is
        # not collected, so resetting it chooses nothing, and treating it as a
        # choice would write the pair's active member to every row.
        if not self._in_inactive_exclusive_choice(name):
            self._mark_edited(name)

    def _build_exclusive_groups(self, visible_names: list[str]) -> None:
        editable_names = [name for name in visible_names if not self._is_locked_name(name)]
        visible_set = set(editable_names)

        adjacency: dict[str, set[str]] = {name: set() for name in editable_names}
        for name in editable_names:
            spec = self._specs_by_name[name]
            if spec.exclusive_group:
                for candidate in editable_names:
                    if candidate == name:
                        continue
                    if self._specs_by_name[candidate].exclusive_group == spec.exclusive_group:
                        adjacency[name].add(candidate)
                        adjacency[candidate].add(name)
            for conflict in spec.mutually_exclusive_with:
                if conflict in visible_set and not self._is_locked_name(conflict):
                    adjacency[name].add(conflict)
                    adjacency[conflict].add(name)

        self._exclusive_groups = {}
        visited: set[str] = set()
        for name in editable_names:
            if name in visited:
                continue
            stack = [name]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adjacency[current] - visited)

            if len(component) <= 1:
                continue

            group_key = tuple(sorted(component, key=lambda item: self._order.get(item, 10_000)))
            self._exclusive_groups[group_key] = set(component)

            selected = self._exclusive_selection.get(group_key)
            if selected not in component:
                selected = self._default_selection_for_group(group_key)
            self._exclusive_selection[group_key] = selected

    def _default_selection_for_group(self, group_key: tuple[str, ...]) -> str:
        # Caller-provided hint takes priority. When both halves of a mutually
        # exclusive pair are pre-filled for UX toggling convenience, the caller
        # uses preferred_exclusive to name the true active choice so that the
        # built-in preference order (kpoint_mp_grid over kpoint_mp_spacing)
        # does not override the user's actual selection.
        hint = self._preferred_exclusive
        if (
            hint is not None
            and hint in group_key
            and hint in self._values
            and gui_value_is_active(self._specs_by_name[hint], self._values[hint])
        ):
            return hint

        for preferred in _PREFERRED_EXCLUSIVE_NAMES:
            if preferred not in group_key or preferred not in self._values:
                continue
            preferred_spec = self._specs_by_name[preferred]
            preferred_value = self._values[preferred]
            if gui_value_is_active(preferred_spec, preferred_value):
                return preferred

        for name in group_key:
            if name not in self._values:
                continue
            spec = self._specs_by_name[name]
            value = self._values[name]
            if gui_value_is_active(spec, value) and not self._is_default_value(spec, value):
                return name

        for name in group_key:
            if name in self._values and gui_value_is_active(
                self._specs_by_name[name], self._values[name]
            ):
                return name

        return group_key[0]

    @staticmethod
    def _is_default_value(spec: KeywordSpec, value: object) -> bool:
        return spec.default_value == value

    def _capture_editor_values(self) -> None:
        for name, row in self._rows.items():
            if row.editor is None:
                continue
            self._values[name] = row.editor.value()

    def _on_row_edited(self, name: str) -> None:
        self._mark_edited(name)
        self._on_editor_changed()

    def _on_editor_changed(self) -> None:
        self._capture_editor_values()

    def _on_exclusive_toggled(self, group_key: tuple[str, ...], name: str, checked: bool) -> None:
        if not checked:
            return

        self._exclusive_selection[group_key] = name
        # Choosing which member of the group is active is a choice for every row.
        self._mark_edited(name)
        for candidate in self._exclusive_groups.get(group_key, set()):
            row = self._rows.get(candidate)
            if row is None or row.editor is None:
                continue
            row.editor.setEnabled(candidate == name)

    def _available_advanced_specs(self) -> list[KeywordSpec]:
        specs: list[KeywordSpec] = []
        for name in sorted(self._specs_by_name, key=lambda item: self._order.get(item, 10_000)):
            if name in self._blocked_names:
                continue
            if not self._is_name_editable(name):
                continue
            if not self._is_advanced_name(name):
                continue
            if name in self._added_advanced:
                continue
            spec = self._specs_by_name[name]
            # Internal keywords stay valid in files but are never offered.
            if not spec.user_visible:
                continue
            specs.append(spec)
        return specs

    def _matches_search(self, spec: KeywordSpec, query: str) -> bool:
        haystacks = [
            spec.name,
            *(spec.aliases or ()),
            spec.label or "",
            spec.description or "",
            spec.category or "",
            keyword_category_label(spec.category),
        ]
        return any(query in candidate.lower() for candidate in haystacks if candidate)

    def _on_task_changed(self) -> None:
        if self._read_only:
            return
        if self._task_editor is None:
            return
        self._task = str(self._task_editor.value()).strip().upper()
        self._prune_incompatible_values()
        self._rebuild_rows()

    def _prune_incompatible_values(self) -> None:
        incompatible = {
            name
            for name, spec in self._specs_by_name.items()
            if not keyword_visible_for_task(spec, task=self._task)
        }
        if not incompatible:
            return
        for name in incompatible:
            self._values.pop(name, None)
            self._added_advanced.discard(name)

    def _open_add_advanced_dialog(self) -> None:
        if self._read_only:
            return
        available = self._available_advanced_specs()
        if not available:
            return

        dialog = _AdvancedSelectionDialog(available, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = set(dialog.selected_names())
        if not selected:
            return

        self._added_advanced.update(selected)
        self._rebuild_rows()

    def _remove_advanced(self, name: str) -> None:
        if self._read_only:
            return
        # Removing the row removes the parameter from every row being edited.
        self._mark_edited(name)
        self._added_advanced.discard(name)
        self._rebuild_rows()

    def _refresh_add_button_state(self) -> None:
        self._add_advanced_button.setEnabled(
            not self._read_only and bool(self._available_advanced_specs())
        )
