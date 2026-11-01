"""The welcome page's entry for one recent project."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

from mufinder.app.eliding_label import ElidingLabel
from mufinder.app.theme import SPACING

_PROJECT_FOLDER_SUFFIX = ".mufinder"
# What the entry asks for when a layout has room to spare; never the path's width.
_PREFERRED_WIDTH = 320


class RecentProjectButton(QPushButton):
    """A button showing a project's name over its elided parent folder.

    The full path is the tooltip. Neither line sets the button's width, so a
    long path cannot widen the card or the window that hold it.
    """

    def __init__(self, project_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        path = Path(project_path)
        # A folder called only ".mufinder" keeps its whole name.
        name = path.name.removesuffix(_PROJECT_FOLDER_SUFFIX) or path.name or project_path

        self._name_label = ElidingLabel(name, self)
        self._name_label.setObjectName("recentProjectName")
        self._folder_label = ElidingLabel(
            str(path.expanduser().absolute().parent), self, elide_mode=Qt.TextElideMode.ElideMiddle
        )
        self._folder_label.setObjectName("recentProjectFolder")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.md, SPACING.xs, SPACING.md, SPACING.xs)
        layout.setSpacing(0)
        for label in (self._name_label, self._folder_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(label)

        # A push button's default ``Minimum`` policy would make the size hint a floor.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setToolTip(project_path)
        self.setAccessibleName(f"Open recent project {name}")
        self.setAccessibleDescription(project_path)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(_PREFERRED_WIDTH, self.minimumSizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return self.layout().minimumSize()
