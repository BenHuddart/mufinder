"""Parameter help dialog sourced from CASTEP keyword metadata."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextEdit, QVBoxLayout, QWidget

from mufinder.app.theme import style_section_title
from mufinder.castep_io.schema import KeywordSpec


class ParameterHelpDialog(QDialog):
    """Read-only dialog describing one CASTEP parameter."""

    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(spec.label or spec.name.upper())
        self.resize(520, 420)

        layout = QVBoxLayout(self)

        title = QLabel(spec.label or spec.name.upper())
        style_section_title(title)
        layout.addWidget(title)

        summary_lines: list[str] = []
        if spec.description:
            summary_lines.append(spec.description)
        summary_lines.append(f"Keyword: {spec.name.upper()}")
        if spec.aliases:
            summary_lines.append("Aliases: " + ", ".join(alias.upper() for alias in spec.aliases))
        summary_lines.append(f"File: {spec.file}")
        summary_lines.append(f"Type: {spec.value_type}")
        if spec.option_type:
            summary_lines.append(f"CASTEP Type: {spec.option_type}")
        if spec.level:
            summary_lines.append(f"Level: {spec.level}")
        if spec.unit:
            summary_lines.append(f"Unit: {spec.unit}")
        if spec.default_value is not None:
            summary_lines.append(f"Default: {spec.default_value}")
        if spec.allowed_values:
            summary_lines.append("Allowed values: " + ", ".join(spec.allowed_values))
        if spec.category:
            summary_lines.append(f"Category: {spec.category}")
        if spec.allowed_tasks:
            summary_lines.append("Allowed tasks: " + ", ".join(spec.allowed_tasks))
        if spec.exclusive_group:
            summary_lines.append(f"Exclusive group: {spec.exclusive_group}")

        summary = QLabel("\n".join(summary_lines))
        summary.setWordWrap(True)
        layout.addWidget(summary)

        body = QTextEdit()
        body.setReadOnly(True)
        body_sections: list[str] = []
        if spec.help_text:
            body_sections.append(spec.help_text)
        if spec.example:
            body_sections.append(f"Example:\n{spec.example}")
        if spec.notes:
            body_sections.append(f"Notes:\n{spec.notes}")
        body.setPlainText("\n\n".join(body_sections).strip())
        layout.addWidget(body)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
