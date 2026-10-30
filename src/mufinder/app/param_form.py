"""Metadata-driven CASTEP parameter form with progressive disclosure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.help_dialog import ParameterHelpDialog
from mufinder.app.param_widgets import ParamEditorBase, create_param_editor
from mufinder.castep_io.gui_metadata import (
    gui_flag_is_unset,
    gui_value_is_active,
    infer_task,
    keyword_visible_for_task,
    keyword_visible_in_workflow,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec


@dataclass(slots=True)
class _FormRow:
    spec: KeywordSpec
    editor: ParamEditorBase
    container: QWidget
    advanced_only: bool


class CastepParamForm(QWidget):
    """Form that renders CASTEP parameter rows from keyword metadata."""

    form_state_changed = Signal(dict)

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        workflow_stage: str,
        include_advanced: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._workflow_stage = workflow_stage
        self._task = infer_task(workflow_stage)
        self._rows: dict[str, _FormRow] = {}
        self._groups: dict[str, QGroupBox] = {}

        self._advanced_toggle = QCheckBox("Show advanced parameters")
        self._advanced_toggle.setChecked(include_advanced)
        self._advanced_toggle.stateChanged.connect(self._on_advanced_toggled)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.addWidget(self._advanced_toggle)

        self._build_rows(specs)
        self._apply_advanced_visibility()

    def _build_rows(self, specs: list[KeywordSpec]) -> None:
        group_forms: dict[str, QFormLayout] = {}

        for spec in specs:
            category = spec.category or "misc"
            if category not in self._groups:
                title = category.replace("_", " ").title()
                group = QGroupBox(title)
                form = QFormLayout(group)
                self._groups[category] = group
                group_forms[category] = form
                self._root_layout.addWidget(group)

            form = group_forms[category]
            editor = create_param_editor(spec, self)
            editor.value_changed.connect(self._emit_state)
            if spec.name == "task":
                editor.value_changed.connect(self._on_task_changed)

            row_widget = QWidget(self)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor)

            info_button = QPushButton("Info")
            info_button.setMaximumWidth(48)
            info_button.clicked.connect(
                lambda _checked=False, row_spec=spec: self._show_help(row_spec)
            )
            row_layout.addWidget(info_button)

            label = spec.label or spec.name.upper()
            form.addRow(QLabel(label), row_widget)

            advanced_only = not keyword_visible_in_workflow(
                spec,
                workflow_stage=self._workflow_stage,
                include_advanced=False,
            )

            self._rows[spec.name] = _FormRow(
                spec=spec,
                editor=editor,
                container=row_widget,
                advanced_only=advanced_only,
            )

        self._root_layout.addStretch(1)

    def _show_help(self, spec: KeywordSpec) -> None:
        dialog = ParameterHelpDialog(spec, self)
        dialog.exec()

    def _on_advanced_toggled(self) -> None:
        self._apply_advanced_visibility()
        self._emit_state()

    def _apply_advanced_visibility(self) -> None:
        show_advanced = self._advanced_toggle.isChecked()
        for row in self._rows.values():
            row.container.setVisible(
                keyword_visible_for_task(row.spec, task=self._task)
                and (show_advanced or not row.advanced_only)
            )

    def _emit_state(self) -> None:
        self.form_state_changed.emit(self.collect_values())

    def collect_values(self) -> dict[str, Any]:
        """Collect structured values for every parameter row.

        An unchecked flag is left out unless it is on by default; then it is
        collected as ``False``, so the workflow default does not turn it back on.
        """

        values: dict[str, Any] = {}
        for name, row in self._rows.items():
            if not self._row_participates(row):
                continue
            value = row.editor.value()
            if gui_flag_is_unset(row.spec, value):
                continue
            values[name] = value
        return values

    def set_values(self, values: dict[str, Any]) -> None:
        """Apply externally supplied values to matching form rows."""

        for name, value in values.items():
            row = self._rows.get(name)
            if row is None:
                continue
            row.editor.set_value(value)
            if name == "task":
                self._task = str(value).strip().upper()
        self._apply_advanced_visibility()

    def validate(self) -> list[str]:
        """Validate field-level and simple cross-field constraints."""

        return validate_gui_values(
            self.collect_values(),
            {name: row.spec for name, row in self._rows.items()},
            task=self._task,
        )

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_toggle.setChecked(visible)

    def advanced_visible(self) -> bool:
        return self._advanced_toggle.isChecked()

    def _row_participates(self, row: _FormRow) -> bool:
        return keyword_visible_for_task(row.spec, task=self._task) and (
            self._advanced_toggle.isChecked() or not row.advanced_only
        )

    def _on_task_changed(self) -> None:
        task_row = self._rows.get("task")
        if task_row is None:
            return
        self._task = str(task_row.editor.value()).strip().upper()
        for name, row in self._rows.items():
            if name == "task":
                continue
            if not keyword_visible_for_task(row.spec, task=self._task) and gui_value_is_active(
                row.spec,
                row.editor.value(),
            ):
                if row.spec.default_value is not None:
                    row.editor.set_value(row.spec.default_value)
        self._apply_advanced_visibility()
        self._emit_state()
