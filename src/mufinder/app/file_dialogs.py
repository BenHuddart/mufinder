"""File dialogs that open in the folder last chosen for the same purpose.

``RememberedFileDialogs`` keeps the last folder in the application settings,
once for every dialog and once per context such as ``open_project``. A dialog
opens in its context's folder, else the last folder of any dialog, else the
home folder; a remembered folder that has gone falls back to its parent.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QWidget

LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"
LAST_DIALOG_DIR_BY_CONTEXT_PREFIX = "ui/last_dialog_dir_by_context"


class RememberedFileDialogs:
    """Open file and folder dialogs where the last one of their context closed."""

    def __init__(self, settings: QSettings, parent: QWidget) -> None:
        self._settings = settings
        self._parent = parent

    @staticmethod
    def _key(context: str | None = None) -> str:
        if context:
            return f"{LAST_DIALOG_DIR_BY_CONTEXT_PREFIX}/{context}"
        return LAST_DIALOG_DIR_KEY

    def start_dir(self, context: str | None = None) -> str:
        """Return the folder a dialog for ``context`` opens in."""

        raw_value = self._settings.value(self._key(context), "", str)
        if not raw_value and context:
            raw_value = self._settings.value(self._key(), "", str)
        if not raw_value:
            return str(Path.home())

        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            return str(candidate if candidate.is_dir() else candidate.parent)

        parent = candidate.parent
        if parent.exists() and parent.is_dir():
            return str(parent)
        return str(Path.home())

    def remember(self, selected_path: str | Path, context: str | None = None) -> None:
        """Remember the folder of ``selected_path`` for every dialog and for ``context``."""

        directory = Path(selected_path).expanduser()
        if directory.exists() and directory.is_file():
            directory = directory.parent
        elif not directory.exists():
            directory = directory.parent

        if directory.exists() and directory.is_dir():
            stored = str(directory)
            self._settings.setValue(self._key(), stored)
            if context:
                self._settings.setValue(self._key(context), stored)

    def get_existing_directory(self, title: str, context: str) -> str:
        """Ask for a folder; return it, or ``""`` when cancelled."""

        selected = QFileDialog.getExistingDirectory(
            self._parent,
            title,
            self.start_dir(context),
        )
        if selected:
            self.remember(selected, context)
        return selected

    def get_open_file_name(self, title: str, file_filter: str, context: str) -> tuple[str, str]:
        """Ask for a file to open; return it and the chosen filter, or ``""`` when cancelled."""

        file_path, selected_filter = QFileDialog.getOpenFileName(
            self._parent,
            title,
            self.start_dir(context),
            file_filter,
        )
        if file_path:
            self.remember(file_path, context)
        return file_path, selected_filter

    def get_save_file_name(self, title: str, file_filter: str, context: str) -> tuple[str, str]:
        """Ask where to save a file; return it and the chosen filter, or ``""`` when cancelled."""

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self._parent,
            title,
            self.start_dir(context),
            file_filter,
        )
        if file_path:
            self.remember(file_path, context)
        return file_path, selected_filter
