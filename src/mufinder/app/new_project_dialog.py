"""The dialog that names a new project and says where it will go."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.eliding_label import ElidingLabel
from mufinder.app.theme import style_warning_text
from mufinder.persistence.json_store import PROJECT_FOLDER_SUFFIX

_MINIMUM_WIDTH = 520
_CHARACTERS_NOT_IN_A_NAME = ("/", "\\", "\0")


class NewProjectDialog(QDialog):
    """Ask for a name and a location, showing the folder they make.

    It creates nothing: the caller passes :meth:`project_name` and
    :meth:`project_root` to ``ProjectService.create_project``. ``Create`` stays
    disabled, with the reason shown, until those make a usable new folder.
    """

    def __init__(self, start_location: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(_MINIMUM_WIDTH)

        self._name_edit = QLineEdit(self)
        self._name_edit.setAccessibleName("Project name")
        self._location_edit = QLineEdit(start_location, self)
        self._location_edit.setAccessibleName("Project location")
        self._browse_button = QPushButton("Browse…", self)
        self._browse_button.setAccessibleName("Browse for the project location")
        self._browse_button.clicked.connect(self._browse)
        location_row = QHBoxLayout()
        location_row.addWidget(self._location_edit, 1)
        location_row.addWidget(self._browse_button)

        # Neither a long location nor a long reason widens the dialog.
        self._target_label = ElidingLabel("", self, elide_mode=Qt.TextElideMode.ElideMiddle)
        self._target_label.setAccessibleName("Project folder to create")

        form = QFormLayout()
        form.addRow("Name", self._name_edit)
        form.addRow("Location", location_row)
        form.addRow("Creates", self._target_label)

        self._reason_label = QLabel(self)
        self._reason_label.setWordWrap(True)
        style_warning_text(self._reason_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._create_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._create_button.setText("Create")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._reason_label)
        layout.addStretch(1)
        layout.addWidget(self._buttons)

        self._name_edit.textChanged.connect(self._refresh)
        self._location_edit.textChanged.connect(self._refresh)
        self._refresh()
        self._name_edit.setFocus()

    def project_name(self) -> str:
        """Return the project's name, without the folder suffix if it was typed."""

        return self._name_edit.text().strip().removesuffix(PROJECT_FOLDER_SUFFIX).strip()

    def project_root(self) -> Path | None:
        """Return the folder to create, or ``None`` until there is a name and a location."""

        name = self.project_name()
        location = self._location_edit.text().strip()
        if not name or not location:
            return None
        try:
            folder = Path(location).expanduser()
        except RuntimeError:
            # A ~user with no home, perhaps part-way through typing one: not a full path.
            folder = Path(location)
        return folder / f"{name}{PROJECT_FOLDER_SUFFIX}"

    def accept(self) -> None:
        # The folder may have filled since the last edit.
        self._refresh()
        if self._create_button.isEnabled():
            super().accept()

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose the project location", self._location_edit.text().strip()
        )
        if selected:
            self._location_edit.setText(selected)

    def _refresh(self) -> None:
        target = self.project_root()
        text = str(target) if target is not None else ""
        self._target_label.setText(text)
        self._target_label.setToolTip(text)
        reason = self._reason_not_to_create()
        self._reason_label.setText(reason)
        self._create_button.setEnabled(not reason)

    def _reason_not_to_create(self) -> str:
        """Say why the current entries make no usable folder; empty if they do."""

        name = self.project_name()
        if not name:
            return "Enter a project name."
        if name in {".", ".."} or any(char in name for char in _CHARACTERS_NOT_IN_A_NAME):
            return "A project name cannot be '.' or '..', or contain a slash or backslash."
        target = self.project_root()
        if target is None:
            return "Choose a location."
        location = target.parent
        if not location.is_absolute():
            return "The location must be a full path."
        try:
            if not location.is_dir():
                return "The location is not an existing folder."
            if not target.exists():
                return ""
            if not target.is_dir():
                return f"{target.name} already exists there and is not a folder."
            if any(target.iterdir()):
                return f"{target.name} already exists there and is not empty."
        except OSError as exc:
            return str(exc)
        return ""
