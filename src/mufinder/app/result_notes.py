"""Notes shown beside magnetic results, shared by the pages and dialogs that show them.

The Magnetic Fields page, its dipolar-tensor and nuclear-relaxation dialogs and
the Polarisation page all mark results calculated at an unrelaxed UEP site with
the same note. This module is light on purpose: a page that only needs the note
does not import the Magnetic Fields page.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget

from mufinder.app.theme import style_hint_text
from mufinder.services.magnetic_field_results import UNRELAXED_RESULTS_NOTE


def unrelaxed_results_note(parent: QWidget) -> QLabel:
    """Return a hidden note marking results computed at an unrelaxed UEP site as estimates."""

    note = QLabel(UNRELAXED_RESULTS_NOTE, parent)
    note.setWordWrap(True)
    style_hint_text(note)
    note.setVisible(False)
    return note
