"""The navigator panel: the splitter that puts the project navigator beside the pages.

:class:`NavigatorPanel` holds the splitter between the
:class:`~mufinder.app.project_navigator.ProjectNavigator` and the page area,
and the navigator's width: the width saved in the application settings, which
a new window opens at, and the width the user drags the splitter's handle to,
which the window saves when it closes. A window too narrow for the saved width
squeezes the navigator without changing what is saved.

It also decides whether the navigator shows: only while a project is open, and
only while ``View > Project Navigator`` (:attr:`NavigatorPanel.toggle_action`,
Ctrl+B) is checked. The user's choice is saved as they make it and restored by
the next window; hiding the navigator keeps its width for when it shows again.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QSplitter, QWidget

from mufinder.app.project_navigator import ProjectNavigator
from mufinder.app.settings_values import read_bool_setting

#: Where the width the user dragged the navigator to is remembered between runs.
WIDTH_SETTINGS_KEY = "ui/project_navigator_width"
#: The widest saved width read back; a larger one falls back to the default.
MAX_SAVED_WIDTH = 4096
#: Where ``View > Project Navigator``'s choice is remembered between runs.
SHOWN_SETTINGS_KEY = "ui/project_navigator_shown"
#: The key that hides and shows the navigator (Cmd+B on macOS).
TOGGLE_SHORTCUT = QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_B)


class NavigatorPanel(QObject):
    """The navigator's splitter beside the page area, its width and whether it shows."""

    def __init__(
        self,
        navigator: ProjectNavigator,
        page_area: QWidget,
        settings: QSettings,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._navigator = navigator
        self._page_area = page_area
        self._settings = settings
        self._project_open = False

        self.splitter = QSplitter(Qt.Orientation.Horizontal, parent)
        self.splitter.setObjectName("projectNavigatorSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        self.splitter.addWidget(navigator)
        self.splitter.addWidget(page_area)
        self.splitter.handle(1).setAccessibleName("Resize project navigator")
        # Window resizes go to the pages; the navigator keeps its width.
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self.saved_width(), parent.width()])
        self.splitter.handle(1).installEventFilter(self)
        self._width_in_drag = navigator.width()
        self._dragged_width: int | None = None
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        self.toggle_action = QAction("&Project Navigator", parent)
        self.toggle_action.setCheckable(True)
        self.toggle_action.setChecked(
            read_bool_setting(self._settings, SHOWN_SETTINGS_KEY, default=True)
        )
        self.toggle_action.setShortcut(TOGGLE_SHORTCUT)
        # A held key toggles once, not on every repeat.
        self.toggle_action.setAutoRepeat(False)
        self.toggle_action.setEnabled(False)
        self.toggle_action.toggled.connect(self._on_toggled)
        # Until the window first calls set_project_open, the navigator is as it
        # will be with a project open, so the window measures its minimum size,
        # which opening a project keeps, with the navigator only if it will show.
        navigator.setVisible(self.toggle_action.isChecked())

    def set_project_open(self, project_open: bool) -> None:
        """Show the navigator if a project is open and the user has not hidden it."""

        self._project_open = project_open
        self.toggle_action.setEnabled(project_open)
        self._sync_visibility()

    def focus_navigator_or_page(self) -> None:
        """Give the navigator keyboard focus, or the page while the navigator is hidden.

        With another window active, such as a dialog still open, this only sets
        what takes focus when the main window is active again.
        """

        if self._navigator.isHidden():
            self._focus_page()
        else:
            self._navigator.focus_list()

    def _on_toggled(self, shown: bool) -> None:
        self._settings.setValue(SHOWN_SETTINGS_KEY, shown)
        if not shown and self._navigator_has_focus():
            # Before hiding, which would otherwise pass focus to whatever is next.
            self._focus_page()
        self._sync_visibility()
        if shown and not self._navigator.isHidden():
            self._navigator.focus_list()

    def _sync_visibility(self) -> None:
        # The one place that decides: a project is open and the user wants it.
        self._navigator.setVisible(self._project_open and self.toggle_action.isChecked())

    def _navigator_has_focus(self) -> bool:
        focused = QApplication.focusWidget()
        return focused is not None and (
            focused is self._navigator or self._navigator.isAncestorOf(focused)
        )

    def _focus_page(self) -> None:
        """Focus the first control of the page shown; leave focus alone if it has none.

        A window not yet on screen takes the control as the one it focuses
        when it is shown.
        """

        window = self._page_area.window()
        widget = window.nextInFocusChain()
        while widget is not window:
            if (
                self._page_area.isAncestorOf(widget)
                and widget.isVisibleTo(window)
                and widget.isEnabled()
                and widget.focusPolicy() & Qt.FocusPolicy.TabFocus
            ):
                widget.setFocus(Qt.FocusReason.OtherFocusReason)
                return
            widget = widget.nextInFocusChain()

    def saved_width(self) -> int:
        """Return the saved width, or the default when none is saved or it is unusable."""

        raw_value = self._settings.value(WIDTH_SETTINGS_KEY, ProjectNavigator.DEFAULT_WIDTH)
        try:
            width = int(raw_value)
        except (TypeError, ValueError):
            return ProjectNavigator.DEFAULT_WIDTH
        if not ProjectNavigator.MINIMUM_WIDTH <= width <= MAX_SAVED_WIDTH:
            return ProjectNavigator.DEFAULT_WIDTH
        return width

    def save_width(self) -> None:
        """Save the width the user dragged the navigator to, if they dragged it."""

        if self._dragged_width is not None:
            self._settings.setValue(WIDTH_SETTINGS_KEY, self._dragged_width)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if event.type() == QEvent.Type.MouseButtonPress and watched is self.splitter.handle(1):
            self._width_in_drag = self._navigator.width()
        return super().eventFilter(watched, event)

    def _on_splitter_moved(self, _position: int, _index: int) -> None:
        # Only a drag lands here, so a window too narrow for the chosen width
        # squeezes the navigator for now without changing the width save_width saves.
        # The signal also fires for a drag the page minimum blocks; that one chose nothing.
        width = self._navigator.width()
        if width == self._width_in_drag:
            return
        self._width_in_drag = width
        self._dragged_width = width
