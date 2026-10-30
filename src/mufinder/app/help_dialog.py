"""Parameter help dialog sourced from CASTEP keyword metadata."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextEdit, QVBoxLayout, QWidget

from mufinder.app.theme import style_section_title
from mufinder.castep_io.gui_metadata import exclusive_partner_labels
from mufinder.castep_io.schema import KeywordSpec


def _plain(text: str | None) -> str:
    """Return catalog text without the backticks it uses to mark keywords."""

    return (text or "").replace("`", "").strip()


def _without_leading(text: str, lead: str) -> str:
    """Return ``text`` without ``lead`` at its start, ignoring how whitespace is laid out.

    A full catalog's one-line description is the first sentence of CASTEP's text,
    with its line breaks joined, so the details leave that sentence out.
    """

    words = lead.split()
    if not words:
        return text
    position = 0
    for word in words:
        while position < len(text) and text[position].isspace():
            position += 1
        if not text.startswith(word, position):
            return text
        position += len(word)
    if position < len(text) and not text[position].isspace():
        return text
    return text[position:].strip()


def _default_text(value: object) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, (tuple, list)):
        return " ".join(str(item) for item in value)
    return str(value)


def exclusive_sentence(partners: tuple[str, ...]) -> str:
    """Say in words which keywords cannot be set together with this one."""

    if not partners:
        return ""
    if len(partners) == 1:
        return f"Use either this or {partners[0]}."
    return f"Use only one of this, {', '.join(partners[:-1])} or {partners[-1]}."


def parameter_help_summary(spec: KeywordSpec) -> list[str]:
    """Return the summary lines the Info dialog shows above the details."""

    lines: list[str] = []
    description = _plain(spec.description)
    if description:
        lines.append(description)
    lines.append(f"CASTEP keyword: {spec.name.upper()}")
    if spec.unit:
        lines.append(f"Unit: {spec.unit}")
    if spec.default_value is not None:
        lines.append(f"Default: {_default_text(spec.default_value)}")
    exclusive = exclusive_sentence(exclusive_partner_labels(spec))
    if exclusive:
        lines.append(exclusive)
    return lines


def parameter_help_details(spec: KeywordSpec) -> str:
    """Return the Info dialog's details: CASTEP's own text, the example and the notes.

    Each text appears once: CASTEP's description is left out when it repeats the
    notes, and neither repeats the one-line description they often start with.
    """

    description = _plain(spec.description)
    notes = _plain(spec.notes)
    help_text = _plain(spec.help_text)
    sections: list[str] = []
    if help_text and help_text != notes:
        castep_text = _without_leading(help_text, description)
        if castep_text:
            sections.append(castep_text)
    if spec.example:
        sections.append(f"Example:\n{spec.example.strip()}")
    notes_text = _without_leading(notes, description)
    if notes_text:
        sections.append(f"Notes:\n{notes_text}")
    return "\n\n".join(sections)


class ParameterHelpDialog(QDialog):
    """Read-only dialog describing one CASTEP parameter for the user."""

    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(spec.label or spec.name.upper())
        self.resize(520, 420)

        layout = QVBoxLayout(self)

        title = QLabel(spec.label or spec.name.upper())
        style_section_title(title)
        layout.addWidget(title)

        summary = QLabel("\n".join(parameter_help_summary(spec)))
        summary.setWordWrap(True)
        layout.addWidget(summary)

        details = parameter_help_details(spec)
        body = QTextEdit()
        body.setReadOnly(True)
        body.setAccessibleName("Details")
        body.setPlainText(details)
        body.setVisible(bool(details))
        layout.addWidget(body, 1)
        if not details:
            layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
