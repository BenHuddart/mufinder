"""Compact badge showing the currently active host structure."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from mufinder.app.theme import style_section_subtitle
from mufinder.domain.models import Structure


class ActiveHostBadge(QWidget):
    """One-line label: 'Using: <structure label>' or 'Using: (no host)'."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("Using: (no host)")
        style_section_subtitle(self._label)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def set_active_host(self, structure: Structure | None) -> None:
        """Update the displayed structure label."""
        if structure is None:
            self._label.setText("Using: (no host)")
        else:
            self._label.setText(f"Using: {structure.label}")
