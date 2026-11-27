"""Numbered steps at the top of a workflow page.

A page that shows its flow lists the steps the user follows, one row each: the
number, a short title, where the step stands, and its actions. The columns line
up across the rows. The page owns what each step says and does.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QWidget

from mufinder.app.theme import form_spacing, style_hint_text, style_section_title


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One step's parts: ``status`` says where it stands, ``actions`` holds its controls."""

    status: QLabel
    actions: QHBoxLayout


class WorkflowSteps(QWidget):
    """The numbered step rows of a workflow page, in the order they are added."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(form_spacing())
        self._grid.setVerticalSpacing(form_spacing())
        self._grid.setColumnStretch(3, 1)
        self._count = 0

    def add_step(self, title: str) -> WorkflowStep:
        """Append the next numbered step called ``title`` and return its parts."""

        row = self._count
        self._count += 1
        centred = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        for column, text in enumerate((str(self._count), title)):
            label = QLabel(text, self)
            style_section_title(label)
            self._grid.addWidget(label, row, column, centred)
        status = QLabel("", self)
        style_hint_text(status)
        self._grid.addWidget(status, row, 2, centred)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(form_spacing())
        self._grid.addLayout(actions, row, 3)
        return WorkflowStep(status=status, actions=actions)
