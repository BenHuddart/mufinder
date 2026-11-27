"""The recently opened projects, kept in the application settings.

``RecentProjects`` holds the list, newest first, and emits ``changed`` whenever
it changes; the File menu's ``Open Recent`` submenu (``RecentProjectsMenu``)
and the welcome page both listen. ``RecentProjects`` asks nothing, so any caller
can clear it; the menu's ``Clear Recent Projects`` confirms first.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from mufinder.app import workflow_table_utils
from mufinder.domain.text import plural

RECENT_PROJECTS_KEY = "ui/recent_projects"
MAX_RECENT_PROJECTS = 10


class RecentProjects(QObject):
    """The recent project paths stored under ``RECENT_PROJECTS_KEY``, newest first."""

    changed = Signal()

    def __init__(self, settings: QSettings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings

    def paths(self) -> list[str]:
        """Return the stored paths, trimmed, without blanks or repeats, at most ten."""

        raw_value = self._settings.value(RECENT_PROJECTS_KEY, [])
        if isinstance(raw_value, str):
            candidates = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            candidates = [str(entry) for entry in raw_value]
        else:
            candidates = []

        deduped: list[str] = []
        seen: set[str] = set()
        for entry in candidates:
            normalized = entry.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped[:MAX_RECENT_PROJECTS]

    def set_paths(self, project_paths: list[str]) -> None:
        """Store ``project_paths`` as the list, dropping blanks and keeping at most ten."""

        cleaned = [path for path in project_paths if path.strip()][:MAX_RECENT_PROJECTS]
        self._settings.setValue(RECENT_PROJECTS_KEY, cleaned)
        self.changed.emit()

    def record(self, project_root: Path) -> None:
        """Put ``project_root`` first, removing any earlier entry for it."""

        normalized = str(project_root.expanduser())
        recent = [entry for entry in self.paths() if entry != normalized]
        recent.insert(0, normalized)
        self.set_paths(recent)

    def remove(self, project_path: str) -> None:
        """Remove ``project_path`` from the list."""

        self.set_paths([entry for entry in self.paths() if entry != project_path])

    def clear(self) -> None:
        """Empty the list."""

        self.set_paths([])


class RecentProjectsMenu(QMenu):
    """``File > Open Recent``: one entry per recent project, then a way to clear them."""

    open_requested = Signal(str)

    def __init__(self, recent_projects: RecentProjects, parent: QWidget | None = None) -> None:
        super().__init__("Open &Recent", parent)
        self._recent_projects = recent_projects
        recent_projects.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the entries from the current list."""

        self.clear()

        recent = self._recent_projects.paths()
        if not recent:
            placeholder = self.addAction("(No recent projects)")
            placeholder.setEnabled(False)
            return

        for project_path in recent:
            # A path is shown as it is: ``&&`` is a literal ampersand, not a mnemonic.
            action = self.addAction(project_path.replace("&", "&&"))
            action.triggered.connect(
                lambda _checked=False, project_path=project_path: self.open_requested.emit(
                    project_path
                )
            )

        self.addSeparator()
        clear_action = self.addAction("&Clear Recent Projects")
        clear_action.triggered.connect(self._confirm_clear)

    def _confirm_clear(self) -> None:
        """Empty the list once the user agrees; no project or file is touched."""

        count = len(self._recent_projects.paths())
        if not count:
            return
        if workflow_table_utils.confirm_destructive(
            self._dialog_parent(),
            "Clear recent projects",
            f"Clear {plural(count, 'recent project')} from the list?",
            "Open Recent and the welcome page will list no projects. "
            "No project or file is deleted.",
            "Clear list",
        ):
            self._recent_projects.clear()

    def _dialog_parent(self) -> QWidget:
        """Return the window the menu belongs to, so the question is centred on it."""

        widget: QWidget = self
        while isinstance(widget, QMenu) and widget.parentWidget() is not None:
            widget = widget.parentWidget()
        if isinstance(widget, QMenu):
            # A menu with no window above it: centre on whichever window is active.
            return QApplication.activeWindow() or widget
        return widget.window()
