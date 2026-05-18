"""Standalone CASTEP parameter dialog with progressive advanced controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.help_dialog import ParameterHelpDialog
from mufinder.app.param_widgets import ParamEditorBase, create_param_editor
from mufinder.castep_io.gui_metadata import (
    gui_value_is_active,
    infer_task,
    keyword_locked_in_workflow,
    keyword_visible_for_task,
    keyword_visible_in_workflow,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec, keyword_catalog_source_info

_ALWAYS_VISIBLE_ADVANCED_NAMES = ("spin_polarized",)
_PREFERRED_EXCLUSIVE_NAMES = ("cut_off_energy",)


@dataclass(slots=True)
class _DialogRow:
    spec: KeywordSpec
    editor: ParamEditorBase | None
    radio: QRadioButton | None
    locked: bool
    advanced: bool


class _AdvancedSelectionDialog(QDialog):
    """Dialog that allows selecting one or more advanced parameters to add."""

    def __init__(self, specs: list[KeywordSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Advanced Parameters")
        self.resize(420, 460)

        layout = QVBoxLayout(self)

        blurb = QLabel("Select one or more advanced parameters to add.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Search by name, alias, description, or category")
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for spec in specs:
            label = spec.label or spec.name.upper()
            text = label if not spec.description else f"{label} - {spec.description}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            item.setData(
                Qt.ItemDataRole.UserRole + 1,
                " ".join(
                    [
                        spec.name,
                        *(spec.aliases or ()),
                        spec.label or "",
                        spec.description or "",
                        spec.category or "",
                    ]
                ).lower(),
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._list.addItem(item)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_names(self) -> tuple[str, ...]:
        selected: list[str] = []
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            name = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(name, str):
                selected.append(name)
        return tuple(selected)

    def _apply_filter(self) -> None:
        query = self._search.text().strip().lower()
        for index in range(self._list.count()):
            item = self._list.item(index)
            haystack = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            item.setHidden(bool(query) and query not in haystack)


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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
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

        root = QVBoxLayout(self)

        catalog_source = keyword_catalog_source_info()
        if catalog_source is not None and catalog_source.origin != "env_json:MUFINDER_CASTEP_KEYWORDS_JSON":
            if not catalog_source.origin.startswith("env_json:"):
                self._catalog_notice = QLabel(
                    "Using fallback CASTEP keyword metadata discovery from "
                    f"`{catalog_source.path}`. Configure a specific keywords JSON in Setup > CASTEP "
                    "if you want to pin the metadata source explicitly."
                )
                self._catalog_notice.setWordWrap(True)
                self._catalog_notice.setObjectName("catalogFallbackNotice")
                root.addWidget(self._catalog_notice)

        if intro_text:
            self._intro_label = QLabel(intro_text, self)
            self._intro_label.setWordWrap(True)
            self._intro_label.setObjectName("parameterDialogIntro")
            root.addWidget(self._intro_label)

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
        self._search_input.textChanged.connect(lambda *_: self._rebuild_rows())
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

        self._rebuild_rows()

    def set_values(self, values: dict[str, Any]) -> None:
        """Apply externally supplied values to known rows."""

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
        self._exclusive_selection.clear()
        self._rebuild_rows(preserve_current=False)

    def collect_values(self) -> dict[str, Any]:
        """Collect structured values for currently active rows."""

        values: dict[str, Any] = {"task": self._task}
        for name, row in self._rows.items():
            if row.editor is None:
                continue
            if self._in_inactive_exclusive_choice(name):
                continue
            value = row.editor.value()
            if value in (None, ""):
                continue
            if row.spec.kind == "flag" and not bool(value):
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
            if name in self._specs_by_name and self._is_advanced_name(name) and name not in self._blocked_names
        }
        for name, value in self._values.items():
            if (
                name in self._specs_by_name
                and self._is_advanced_name(name)
                and name not in self._blocked_names
                and gui_value_is_active(self._specs_by_name[name], value)
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
            or keyword_locked_in_workflow(self._specs_by_name[name], workflow_stage=self._workflow_stage)
        }

        blocked = set()
        for name, spec in self._specs_by_name.items():
            if name in locked_names:
                continue
            if any(conflict in locked_names for conflict in spec.mutually_exclusive_with):
                blocked.add(name)
        return blocked

    def _visible_names(self) -> list[str]:
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

        query = self._search_input.text().strip().lower()
        if query:
            names = [name for name in names if self._matches_search(self._specs_by_name[name], query)]

        names.sort(key=lambda item: self._order.get(item, 10_000))
        return names

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

        for row_index, name in enumerate(visible_names, start=1):
            spec = self._specs_by_name[name]
            group_key = self._group_key_for_name(name)

            radio: QRadioButton | None = None
            label_text = spec.label or spec.name.upper()
            if group_key is not None:
                label_widget = QWidget(self._table_widget)
                label_layout = QHBoxLayout(label_widget)
                label_layout.setContentsMargins(0, 0, 0, 0)
                radio = QRadioButton(label_widget)
                radio.setChecked(self._exclusive_selection.get(group_key) == name)
                radio.setEnabled(self._is_name_editable(name))
                radio.toggled.connect(
                    lambda checked, g=group_key, n=name: self._on_exclusive_toggled(g, n, checked)
                )
                label_layout.addWidget(radio)
                label_layout.addWidget(QLabel(label_text, label_widget))
                label_layout.addStretch(1)
            else:
                label_widget = QLabel(label_text, self._table_widget)

            self._table_layout.addWidget(label_widget, row_index, 0)

            locked = self._is_locked_name(name)
            editor: ParamEditorBase | None = None
            if locked:
                value_text = self._locked_values.get(name, "Variable (set by workflow)")
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
                editor.value_changed.connect(self._on_editor_changed)
                if self._in_inactive_exclusive_choice(name):
                    editor.setEnabled(False)
                elif not self._is_name_editable(name):
                    editor.setEnabled(False)
                value_widget = editor

            self._table_layout.addWidget(value_widget, row_index, 1)

            reset_button = QPushButton("Reset", self._table_widget)
            reset_button.setMaximumWidth(58)
            reset_button.setEnabled(self._is_name_editable(name) and not locked and editor is not None)
            if self._is_name_editable(name) and not locked and editor is not None:
                reset_button.clicked.connect(lambda _checked=False, param=name: self._reset_parameter(param))
            self._table_layout.addWidget(reset_button, row_index, 2)

            info_button = QPushButton("Info", self._table_widget)
            info_button.setMaximumWidth(52)
            info_button.clicked.connect(lambda _checked=False, row_spec=spec: self._show_help(row_spec))
            self._table_layout.addWidget(info_button, row_index, 3)

            remove_button = QToolButton(self._table_widget)
            remove_button.setText("x")
            can_remove = (
                self._is_name_editable(name)
                and self._is_advanced_name(name)
                and name in self._added_advanced
                and not locked
            )
            remove_button.setEnabled(can_remove)
            remove_button.setVisible(can_remove)
            if can_remove:
                remove_button.clicked.connect(lambda _checked=False, param=name: self._remove_advanced(param))
            self._table_layout.addWidget(remove_button, row_index, 4)

            self._rows[name] = _DialogRow(
                spec=spec,
                editor=editor,
                radio=radio,
                locked=locked,
                advanced=self._is_advanced_name(name),
            )

        self._table_layout.setRowStretch(len(visible_names) + 1, 1)
        self._refresh_add_button_state()

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
            if name in self._values and gui_value_is_active(self._specs_by_name[name], self._values[name]):
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

    def _on_editor_changed(self) -> None:
        self._capture_editor_values()

    def _on_exclusive_toggled(self, group_key: tuple[str, ...], name: str, checked: bool) -> None:
        if not checked:
            return

        self._exclusive_selection[group_key] = name
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
            specs.append(self._specs_by_name[name])
        return specs

    def _matches_search(self, spec: KeywordSpec, query: str) -> bool:
        haystacks = [
            spec.name,
            *(spec.aliases or ()),
            spec.label or "",
            spec.description or "",
            spec.category or "",
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
            name for name, spec in self._specs_by_name.items() if not keyword_visible_for_task(spec, task=self._task)
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
        self._added_advanced.discard(name)
        self._rebuild_rows()

    def _refresh_add_button_state(self) -> None:
        self._add_advanced_button.setEnabled(not self._read_only and bool(self._available_advanced_specs()))
