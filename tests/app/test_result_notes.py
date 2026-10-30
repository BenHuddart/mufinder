"""The shared note beside magnetic results calculated at an unrelaxed site."""

from __future__ import annotations

import subprocess
import sys

from PySide6.QtWidgets import QWidget

from mufinder.app.result_notes import unrelaxed_results_note
from mufinder.services.magnetic_field_results import UNRELAXED_RESULTS_NOTE


def test_the_note_starts_hidden_and_wraps(qapp) -> None:
    parent = QWidget()

    note = unrelaxed_results_note(parent)

    assert note.parent() is parent
    assert note.text() == UNRELAXED_RESULTS_NOTE
    assert note.wordWrap()
    assert note.isHidden()
    assert note.objectName() == "hintText"
    parent.close()


def test_the_polarisation_page_does_not_load_the_magnetic_fields_page() -> None:
    code = (
        "import sys\n"
        "import mufinder.app.polarisation_widget\n"
        "assert 'mufinder.app.magnetic_field_widget' not in sys.modules, 'loaded'\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
