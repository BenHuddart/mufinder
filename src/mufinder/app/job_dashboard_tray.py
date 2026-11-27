"""The Job Dashboard tray: the dock at the bottom of the main window that holds the dashboard.

The tray is a ``QDockWidget`` that docks at the bottom only, never floats and
has no close button. Its title bar is :class:`JobDashboardHeader`, a strip with
the collapse toggle, a summary of the listed jobs by state whose counts filter
the table by state (:class:`JobSummary`), the dashboard's selection hint and
``Show dismissed``. :class:`JobDashboardTray` keeps it in one
of two states:

- collapsed: the strip alone, which cannot be dragged taller;
- expanded: as tall as the table needs to show every row, at most
  :data:`TRAY_HEIGHT_SHARE` of the height the tray shares with the page area,
  the table scrolling beyond that. A height the user drags the tray's edge to
  is kept until the tray is collapsed or expanded again.

When the rows need more height than the page's minimum leaves, the page is
clipped rather than covered: the page stack sits in :class:`ClippablePageArea`,
which gives up its minimum height while the tray needs the room. The window's
own minimum size stays what it is with the tray collapsed, so collapsing never
resizes the window.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QObject,
    QPoint,
    QRect,
    QSettings,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QWidget,
)

from mufinder.app.icons import icon, job_state_icon
from mufinder.app.job_dashboard import JobDashboard
from mufinder.app.settings_values import read_bool_setting
from mufinder.app.theme import SPACING, refresh_widget_style
from mufinder.domain.text import plural

#: Where the tray's collapsed state is remembered between runs.
COLLAPSED_SETTINGS_KEY = "ui/job_dashboard_collapsed"
#: The key that collapses and expands the tray (Cmd+J on macOS).
TOGGLE_SHORTCUT = QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_J)
#: The most of the height shared with the page area an expanded tray takes.
TRAY_HEIGHT_SHARE = 0.7
#: ``QWidget``'s "no maximum".
_NO_MAXIMUM = (1 << 24) - 1
_STATE_ICON_SIZE = 16


def _joined(parts: Sequence[str]) -> str:
    """Return ``parts`` read as a list: ``a``, ``a and b``, ``a, b and c``."""

    if len(parts) < 2:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def job_summary_sentence(
    counts: Sequence[tuple[str, int]], state_filter: Sequence[str] = ()
) -> str:
    """Return the summary read as one phrase: ``Jobs: 4 running, 12 queued and 2 failed``.

    With a ``state_filter`` it says what the table lists as well:
    ``Jobs: 4 running and 2 failed. Listing failed only.``
    """

    parts = [f"{count} {word}" for word, count in counts]
    if not parts:
        sentence = "No jobs"
    else:
        sentence = f"Jobs: {_joined(parts)}"
    if state_filter:
        sentence += f". Listing {_joined(list(state_filter))} only."
    return sentence


class _PageHolder(QWidget):
    """Hold the page at the holder's size, reporting the page's minimum and size hints.

    It has no layout, so a page whose layout trades height for width is laid
    out at its minimum height, as it would be directly in a splitter, rather
    than at the taller height a scroll area would give it.
    """

    def __init__(self, page: QWidget) -> None:
        super().__init__()
        self.page = page
        page.setParent(self)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        hint = self.page.minimumSizeHint()
        explicit = self.page.minimumSize()
        return QSize(
            explicit.width() or max(hint.width(), 0), explicit.height() or max(hint.height(), 0)
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return self.page.sizeHint().expandedTo(self.minimumSizeHint())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self.page.setGeometry(self.rect())
        super().resizeEvent(event)


class ClippablePageArea(QScrollArea):
    """The page stack's frame, which can let the page be shorter than its minimum.

    It reports the page's minimum size as its own, so the layout around it
    behaves as if the page sat there directly, until
    :meth:`set_height_released` lets it be shorter. The page then keeps its
    minimum height and its lower part is cut off. The page never scrolls: there
    are no scroll bars, and the wheel, the keyboard and focus leave it at its top,
    so collapsing the tray is the one way to see the rest
    (:meth:`hides` tells the tray when focus has gone there). The minimum width
    is never given up.
    """

    #: The page's minimum size may have changed (a page switch, new content).
    page_minimum_changed = Signal()

    def __init__(self, page: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pageClipArea")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._height_released = False
        self._holder = _PageHolder(page)
        self.setWidget(self._holder)
        page.show()
        # The page's minimum changes as pages come and go; this area's follows.
        page.installEventFilter(self)
        # Whatever moves a scroll bar (focus on a hidden control, a key), the page stays put.
        for bar in (self.verticalScrollBar(), self.horizontalScrollBar()):
            bar.valueChanged.connect(lambda value, bar=bar: value and bar.setValue(0))

    def page(self) -> QWidget:
        """Return the page this area frames."""

        return self._holder.page

    def page_minimum_size(self) -> QSize:
        """Return the page's minimum size: its own where set, else its layout's."""

        return self._holder.minimumSizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        minimum = self.page_minimum_size()
        return QSize(minimum.width(), 0 if self._height_released else minimum.height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return self._holder.sizeHint()

    def height_given_up(self) -> int:
        """Return how much the minimum height of the layout around the area falls on release.

        Beside the area in a side-by-side splitter, the tallest minimum among the
        other panels still holds, so only the part of the page's minimum above it
        is given up.
        """

        if not self._height_released:
            return 0
        page_minimum = self.page_minimum_size().height()
        beside = 0
        parent = self.parentWidget()
        if isinstance(parent, QSplitter) and parent.orientation() == Qt.Orientation.Horizontal:
            for index in range(parent.count()):
                panel = parent.widget(index)
                if panel is self or panel.isHidden():
                    continue
                minimum = panel.minimumHeight() or panel.minimumSizeHint().height()
                beside = max(beside, minimum)
        return max(page_minimum, beside) - beside

    def is_height_released(self) -> bool:
        """Return whether the area may be shorter than the page's minimum height."""

        return self._height_released

    def set_height_released(self, released: bool) -> None:
        """Let the area be shorter than the page's minimum height, or give the page it back.

        Given back, the page is scrolled to its top.
        """

        if released == self._height_released:
            return
        self._height_released = released
        self.updateGeometry()
        if not released:
            self.verticalScrollBar().setValue(0)

    def hides(self, widget: QWidget) -> bool:
        """Return whether ``widget`` is in the page and wholly below what the area shows."""

        page = self._holder.page
        if not self.is_clipped() or not page.isAncestorOf(widget):
            return False
        # A control scrolled out of a table or other view inside the page is
        # hidden by that view, not by the area.
        ancestor = widget.parentWidget()
        while ancestor is not None and ancestor is not page:
            in_ancestor = QRect(widget.mapTo(ancestor, QPoint(0, 0)), widget.size())
            if not ancestor.rect().intersects(in_ancestor):
                return False
            ancestor = ancestor.parentWidget()
        top = widget.mapTo(page, QPoint(0, 0)).y()
        return top >= self.viewport().height()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # Not scrolled, and not passed on: a wheel over the page moves nothing.
        event.accept()

    def is_clipped(self) -> bool:
        """Return whether part of the page is cut off below the area."""

        return self._height_released and self._holder.height() > self.viewport().height()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self._holder.page and event.type() == QEvent.Type.LayoutRequest:
            self._holder.updateGeometry()
            self.updateGeometry()
            # The area sizes the holder again on a layout request of its own.
            QCoreApplication.postEvent(self, QEvent(QEvent.Type.LayoutRequest))
            self.page_minimum_changed.emit()
        return super().eventFilter(watched, event)


class JobSummary(QWidget):
    """The listed jobs counted by state, each count a toggle that filters the table by its state.

    Each count is a checkable button with its state's glyph, the count and the
    state's word (``2 failed``); ``No jobs`` shows when the table lists none.
    Ticking or unticking a count asks for a new filter through
    ``filter_requested``; ``Show all``, shown while any count is ticked, asks to
    clear it. The summary shows the filter it is given (:meth:`set_filter`), so
    it never ticks a count by itself. Each state keeps its button while it has a
    count, so focus and the ticked state survive a change of count. The
    summary's accessible name and tooltip read every count as one phrase
    (:func:`job_summary_sentence`).
    """

    #: The states to filter the table by, ``(word, ...)``; empty lists every job.
    filter_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("jobDashboardSummary")
        # Squeezed rather than widening the window when there are many states.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(SPACING.sm, 0, 0, 0)
        self._layout.setSpacing(SPACING.xs)
        self._counts: tuple[tuple[str, int], ...] = ()
        self._filter: tuple[str, ...] = ()
        self._buttons: dict[str, QToolButton] = {}
        # The controls Tab reaches just before and just after the summary's buttons.
        self._tab_before: QWidget | None = None
        self._tab_after: QWidget | None = None

        self._empty = QLabel("No jobs", self)
        self._empty.setObjectName("jobSummaryEmpty")
        self.show_all_button = QToolButton(self)
        self.show_all_button.setObjectName("jobSummaryShowAll")
        self.show_all_button.setText("Show all")
        self.show_all_button.setAccessibleName("Show all jobs")
        self.show_all_button.setToolTip("Clear the filter and list every job.")
        self.show_all_button.clicked.connect(lambda _checked=False: self.filter_requested.emit(()))
        self._sync((), ())

    # ----------------------------------------------------------------- reading

    def counts(self) -> tuple[tuple[str, int], ...]:
        """Return the ``(word, count)`` pairs shown."""

        return self._counts

    def state_filter(self) -> tuple[str, ...]:
        """Return the states shown ticked."""

        return self._filter

    def entry_texts(self) -> list[str]:
        """Return the text of each entry as shown (``4 running``, or ``No jobs``)."""

        if not self._counts:
            return [self._empty.text()]
        return [self._buttons[word].text() for word, _count in self._counts]

    def count_button(self, word: str) -> QToolButton | None:
        """Return the button of the state ``word``, or ``None`` while it has no count shown."""

        return self._buttons.get(word)

    def count_buttons(self) -> list[QToolButton]:
        """Return the count buttons from left to right."""

        return [self._buttons[word] for word, _count in self._counts]

    # ----------------------------------------------------------------- showing

    def set_counts(self, counts: Sequence[tuple[str, int]]) -> None:
        """Show ``counts``, the ``(word, count)`` pairs of the listed jobs in order."""

        counts = tuple((str(word), int(count)) for word, count in counts)
        if counts != self._counts:
            self._sync(counts, self._filter)

    def set_filter(self, words: Sequence[str]) -> None:
        """Show the states in ``words`` ticked, and ``Show all`` while there are any."""

        words = tuple(str(word) for word in words)
        if words != self._filter:
            self._sync(self._counts, words)

    def set_tab_neighbours(self, before: QWidget, after: QWidget) -> None:
        """Put the buttons in the Tab order between ``before`` and ``after``, left to right."""

        self._tab_before = before
        self._tab_after = after
        self._apply_tab_order()

    def _strip_controls(self) -> list[QWidget]:
        """Return the summary's controls that Tab can reach now, left to right."""

        controls: list[QWidget] = self.count_buttons()
        if not self.show_all_button.isHidden():
            controls.append(self.show_all_button)
        return controls

    def _sync(self, counts: tuple[tuple[str, int], ...], words: tuple[str, ...]) -> None:
        """Show ``counts`` with the states in ``words`` ticked, keeping each state's button."""

        before = self._strip_controls()
        focus = QApplication.focusWidget()
        focus_index = before.index(focus) if focus in before else None
        self._counts = counts
        self._filter = words

        words = [word for word, _count in self._counts]
        for word in [word for word in self._buttons if word not in words]:
            button = self._buttons.pop(word)
            button.hide()
            button.deleteLater()
        while self._layout.takeAt(0) is not None:
            pass
        self._empty.setVisible(not self._counts)
        self._layout.addWidget(self._empty)
        for word, count in self._counts:
            button = self._buttons.get(word)
            if button is None:
                button = self._new_button(word)
                self._buttons[word] = button
            self._show_count(button, word, count)
            self._layout.addWidget(button)
        self.show_all_button.setVisible(bool(self._filter))
        self._layout.addWidget(self.show_all_button)
        self._layout.addStretch(1)

        sentence = job_summary_sentence(self._counts, self._filter)
        self.setAccessibleName(sentence)
        # The strip cuts a long summary short; the tooltip always has every count.
        self.setToolTip(sentence)
        self._apply_tab_order()
        if focus_index is not None:
            self._keep_focus(before, focus_index)

    def _keep_focus(self, before: list[QWidget], focus_index: int) -> None:
        """Give focus to the nearest control left in the strip when the focused one went.

        The nearest is the next one to its right, else to its left, else the
        control before the summary in the Tab order.
        """

        after = self._strip_controls()
        if before[focus_index] in after:
            return
        right = [control for control in before[focus_index + 1 :] if control in after]
        left = [control for control in reversed(before[:focus_index]) if control in after]
        target = (right or left or [self._tab_before])[0]
        if target is not None:
            target.setFocus(Qt.FocusReason.OtherFocusReason)

    def _apply_tab_order(self) -> None:
        if self._tab_before is None or self._tab_after is None:
            return
        chain = [self._tab_before, *self.count_buttons(), self.show_all_button, self._tab_after]
        for first, second in zip(chain, chain[1:], strict=False):
            QWidget.setTabOrder(first, second)

    def _new_button(self, word: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("jobSummaryCount")
        button.setProperty("jobState", word)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Painted when shown, so the glyph follows the theme.
        button.setIcon(job_state_icon(word))
        button.setIconSize(QSize(_STATE_ICON_SIZE, _STATE_ICON_SIZE))
        button.clicked.connect(lambda checked, word=word: self._on_count_clicked(word, checked))
        return button

    def _show_count(self, button: QToolButton, word: str, count: int) -> None:
        ticked = word in self._filter
        button.setText(f"{count} {word}")
        button.setAccessibleName(plural(count, f"{word} job"))
        if ticked:
            button.setToolTip(f"Stop filtering by {word} jobs.")
        else:
            button.setToolTip(
                f"List only the {word} jobs. Tick more counts to list several states."
            )
        if button.isChecked() != ticked:
            with QSignalBlocker(button):
                button.setChecked(ticked)

    def _on_count_clicked(self, word: str, checked: bool) -> None:
        if checked:
            words = (*self._filter, word)
        else:
            words = tuple(ticked for ticked in self._filter if ticked != word)
        self.filter_requested.emit(words)
        # The button shows the filter it is given, whether or not the request took.
        button = self._buttons.get(word)
        if button is not None and button.isChecked() != (word in self._filter):
            with QSignalBlocker(button):
                button.setChecked(word in self._filter)


class JobDashboardHeader(QFrame):
    """The tray's header strip: the toggle, the job summary, the selection hint, ``Show dismissed``.

    It is the dock's title bar. The dashboard keeps owning its selection hint and
    ``Show dismissed`` box; the strip only lays them out.
    """

    def __init__(self, dashboard: JobDashboard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("jobDashboardHeader")
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("jobDashboardToggle")
        self.toggle_button.setText("Job Dashboard")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setAutoRaise(True)
        self.summary = JobSummary(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.xs, SPACING.xs, SPACING.sm, SPACING.xs)
        layout.setSpacing(SPACING.sm)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.summary, 1)
        layout.addWidget(dashboard.selection_hint)
        layout.addWidget(dashboard.show_dismissed_box)


class JobDashboardTray(QObject):
    """Build the Job Dashboard dock and keep it collapsed or fitted to its rows.

    ``window`` is the main window the dock goes in, ``page_area`` the frame of
    its page stack and ``settings`` where the collapsed state is kept.
    """

    def __init__(
        self,
        window: QMainWindow,
        dashboard: JobDashboard,
        page_area: ClippablePageArea,
        settings: QSettings,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._dashboard = dashboard
        self._page_area = page_area
        self._settings = settings
        self._collapsed = False
        # The user dragged the tray's edge since it was last collapsed or expanded.
        self._manual_height = False
        # The dock's height when a press on the window's own area (a dock edge) began.
        self._press_height: int | None = None
        # The window's minimum height is held explicitly while the page gives up its own.
        self._holding_height = False
        # Whether the dock is seen: shown, not minimised and not covered.
        self._dock_seen = True

        self.header = JobDashboardHeader(dashboard)
        dock = QDockWidget("Job Dashboard", window)
        dock.setObjectName("jobDashboardDock")
        # Neither movable nor floatable. Closable only so View > Job Dashboard can
        # hide it (the toggle action closes the dock); the header strip, as its
        # title bar, has no close button.
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setTitleBarWidget(self.header)
        dock.setWidget(dashboard)
        # View > Job Dashboard; its text follows the dock's title, which has no mnemonic.
        dock.toggleViewAction().setText("&Job Dashboard")
        dashboard.bind_delete_keys(dock)
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.dock = dock
        # Tab goes through the strip, left to right, before the table under it:
        # the toggle, each count, ``Show all`` while it shows, ``Show dismissed``.
        toggle = self.header.toggle_button
        before_table = dashboard.table.previousInFocusChain()
        if before_table is not toggle:
            QWidget.setTabOrder(before_table, toggle)
        self.header.summary.set_tab_neighbours(toggle, dashboard.show_dismissed_box)

        self.toggle_shortcut = QShortcut(TOGGLE_SHORTCUT, window)
        self.toggle_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.toggle_shortcut.activated.connect(lambda: self.toggle(from_keyboard=True))
        self.header.toggle_button.clicked.connect(lambda _checked=False: self.toggle())

        summary = self.header.summary
        summary.set_filter(dashboard.state_filter())
        dashboard.state_filter_changed.connect(summary.set_filter)
        dashboard.summary_changed.connect(summary.set_counts)
        summary.filter_requested.connect(self._on_filter_requested)
        dashboard.rows_height_changed.connect(self._on_rows_height_changed)
        # A new page minimum changes the window minimum held while the page is clipped.
        page_area.page_minimum_changed.connect(self._on_page_minimum_changed)
        dashboard.action_requested.connect(self._on_dashboard_action)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

        # Window resizes and dock-edge drags arrive several at a time; one fit follows them.
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(0)
        self._refit_timer.timeout.connect(self._on_refit_due)
        window.installEventFilter(self)
        self._sync_toggle()

    # ------------------------------------------------------------------ state

    def is_collapsed(self) -> bool:
        """Return whether the tray shows its header strip alone."""

        return self._collapsed

    def toggle(self, *, from_keyboard: bool = False) -> None:
        """Collapse an expanded tray or expand a collapsed one; nothing while it is closed."""

        if self.dock.isHidden():
            return
        self.set_collapsed(not self._collapsed, from_keyboard=from_keyboard)

    def set_collapsed(self, collapsed: bool, *, from_keyboard: bool = False) -> None:
        """Collapse the tray to its strip, or expand it to fit its rows, and remember which.

        ``from_keyboard`` moves focus into the table on expanding. Collapsing
        clears the selection, and moves focus out of the table to the toggle.
        """

        if collapsed == self._collapsed:
            return
        self._settings.setValue(COLLAPSED_SETTINGS_KEY, collapsed)
        self._apply_collapsed(collapsed)
        if not collapsed and from_keyboard:
            self._dashboard.table.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def place(self) -> None:
        """Dock the tray at the bottom and give it the remembered state.

        The window calls this once, after restoring its saved state, which may
        hold a dock left floating or at the top.
        """

        if self.dock.isFloating():
            self.dock.setFloating(False)
        if self._window.dockWidgetArea(self.dock) != Qt.DockWidgetArea.BottomDockWidgetArea:
            self._window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        self._apply_collapsed(read_bool_setting(self._settings, COLLAPSED_SETTINGS_KEY))
        # The dock has its real geometry once this show has been laid out.
        self._refit_timer.start()

    def set_dock_seen(self, seen: bool) -> None:
        """Follow the dock's visibility; the table is seen while the dock is and is expanded."""

        self._dock_seen = bool(seen)
        self._dashboard.set_visible_to_user(self._dock_seen, rows_shown=not self._collapsed)
        if self.dock.isHidden():
            self._release_page_height(False)
        elif self._dock_seen:
            self._refit_timer.start()

    def _apply_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._manual_height = False
        table = self._dashboard.table
        if collapsed:
            focus = QApplication.focusWidget()
            focus_inside = focus is not None and self._dashboard.isAncestorOf(focus)
            table.selectionModel().clear()
            table.hide()
            if focus_inside:
                self.header.toggle_button.setFocus(Qt.FocusReason.OtherFocusReason)
            strip = self.dock.minimumSizeHint().height()
            self.dock.setMaximumHeight(strip)
            self._release_page_height(False)
            self._window.resizeDocks([self.dock], [strip], Qt.Orientation.Vertical)
        else:
            table.show()
            self.dock.setMaximumHeight(_NO_MAXIMUM)
            self.refit()
        self._dashboard.set_visible_to_user(self._dock_seen, rows_shown=not collapsed)
        self._sync_toggle()

    def _sync_toggle(self) -> None:
        button = self.header.toggle_button
        shortcut = TOGGLE_SHORTCUT.toString(QKeySequence.SequenceFormat.NativeText)
        if self._collapsed:
            button.setIcon(icon("chevron_up"))
            button.setAccessibleName("Expand Job Dashboard")
            button.setToolTip(f"Expand the Job Dashboard to show its jobs ({shortcut}).")
        else:
            button.setIcon(icon("chevron_down"))
            button.setAccessibleName("Collapse Job Dashboard")
            button.setToolTip(f"Collapse the Job Dashboard to this strip ({shortcut}).")
        self.header.setProperty("collapsed", "true" if self._collapsed else "false")
        refresh_widget_style(self.header)

    # ---------------------------------------------------------------- fitting

    def fitted_height(self) -> int:
        """Return the dock height that shows every listed row, the header strip included."""

        return self.dock.minimumSizeHint().height() + self._dashboard.rows_height_hint()

    def height_cap(self) -> int:
        """Return the most an expanded tray takes: a share of the height it shares with the page."""

        shared = self._shared_height()
        return max(self.dock.minimumSizeHint().height(), int(shared * TRAY_HEIGHT_SHARE))

    def refit(self) -> None:
        """Fit the expanded tray to its rows, under the cap, clipping the page if it must.

        After a drag of its edge the tray keeps the dragged height, held under the cap,
        and while its rows need the room it can still be dragged up to the cap.
        """

        self._refit_timer.stop()
        if self._collapsed or self.dock.isHidden() or not self._window.isVisible():
            return
        cap = self.height_cap()
        self.dock.setMaximumHeight(cap)
        fitted = min(self.fitted_height(), cap)
        target = min(self.dock.height(), cap) if self._manual_height else fitted
        # While the rows need more than the page leaves, the page stays free to be
        # clipped even under a shorter dragged height, so a drag can take the tray
        # back up to its fit: Qt holds a drag to the page's minimum as it happens.
        newly_released = self._release_page_height(
            max(target, fitted) > self._room_without_clipping()
        )
        self._window.resizeDocks([self.dock], [target], Qt.Orientation.Vertical)
        if newly_released:
            # The page's freed minimum can take more than one pass of the event
            # loop to reach the window's layout (as when the tray is shown
            # again), and a resize meanwhile stops at the old minimum.
            self._refit_timer.start()
        self._keep_focus_in_sight()

    def _keep_focus_in_sight(self) -> None:
        """Move focus to the toggle when the tray has come to hide the focused control."""

        focus = QApplication.focusWidget()
        if focus is not None and self._page_area.hides(focus):
            self.header.toggle_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _shared_height(self) -> int:
        central = self._window.centralWidget()
        return (central.height() if central is not None else 0) + self.dock.height()

    def _room_without_clipping(self) -> int:
        """Return the tallest the dock can be while the page keeps its minimum height."""

        central = self._window.centralWidget()
        around_page = (central.height() if central is not None else 0) - self._page_area.height()
        page_minimum = self._page_area.page_minimum_size().height()
        return self._shared_height() - max(around_page, 0) - page_minimum

    def _release_page_height(self, released: bool) -> bool:
        """Let the page area be shorter than the page, or give the page its height back.

        While it is released the window's minimum height is held at what the
        layout gives with the page not released (:meth:`_hold_minimum_height`), so
        it is the same with the tray collapsed. Return whether the page was
        released by this call.
        """

        area = self._page_area
        if released:
            newly_released = not area.is_height_released()
            if newly_released:
                area.set_height_released(True)
            self._hold_minimum_height()
            return newly_released
        if not area.is_height_released():
            return False
        area.set_height_released(False)
        if self._holding_height:
            # With a project open the window's layout sets its minimum again.
            self._holding_height = False
            self._window.setMinimumSize(0, 0)
            self._activate_window_layout()
        return False

    def _hold_minimum_height(self) -> None:
        """Hold the window's minimum height at what it is with the page's minimum kept.

        Only the height is held: the window's layout keeps setting the minimum
        width, which releasing the page never changes. The height is worked out
        from the layout as it is now, with the page area's minimum released, plus
        the height the area gave up, so it follows page switches and any other
        change to the window's layout. It is set again on each fit and whenever
        the page's minimum changes.
        """

        window = self._window
        # Let every pending layout change, the release among them, reach the
        # window's layout before it is read. Each widget between the page area
        # and the window keeps its size hints cached until told they changed.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest.value)
        widget = self._page_area
        while widget is not None and widget is not window:
            widget.updateGeometry()
            if widget.layout() is not None:
                widget.layout().invalidate()
            widget = widget.parentWidget()
        layout = window.layout()
        layout.invalidate()
        height = layout.totalMinimumSize().height() + self._page_area.height_given_up()
        if not self._holding_height or window.minimumHeight() != height:
            # A zero width is not explicit, so the layout goes on setting the width.
            window.setMinimumSize(0, height)
            self._holding_height = True
        self._activate_window_layout()

    def _activate_window_layout(self) -> None:
        layout = self._window.layout()
        layout.invalidate()
        layout.activate()

    # ----------------------------------------------------------------- events

    def _on_refit_due(self) -> None:
        # Not under the pointer: a drag of the tray's edge ends with a fit of its own.
        if self._press_height is None:
            self.refit()

    def _on_rows_height_changed(self) -> None:
        # A dragged height is kept, but whether the page must give up room changes.
        self._refit_timer.start()

    def _on_page_minimum_changed(self) -> None:
        if self._page_area.is_height_released():
            self._refit_timer.start()

    def _on_filter_requested(self, words: Sequence[str]) -> None:
        # Ticking a count asks to see those jobs, so a collapsed tray expands;
        # unticking and Show all leave it as it is.
        ticked = set(words) - set(self._dashboard.state_filter())
        self._dashboard.set_state_filter(words)
        if ticked and self._collapsed:
            self.set_collapsed(False)

    def _on_dashboard_action(self, kind: str, _job_ids: object) -> None:
        # Opening a job's page shows the page: all of it, if the tray was cutting it off.
        if kind == "open" and self._page_area.is_clipped():
            self.set_collapsed(True)

    def _on_focus_changed(self, _old: QWidget | None, new: QWidget | None) -> None:
        # The page does not scroll, so focus on a control the tray hides shows it
        # by collapsing the tray, as opening a job's page does.
        if new is not None and not self._collapsed and self._page_area.hides(new):
            self.set_collapsed(True)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self._window:
            kind = event.type()
            if kind == QEvent.Type.Resize:
                self._refit_timer.start()
            elif kind == QEvent.Type.MouseButtonPress:
                # A press on the window itself, not on a child: a dock edge.
                self._press_height = self.dock.height()
            elif kind == QEvent.Type.MouseButtonRelease and self._press_height is not None:
                dragged = self.dock.height() != self._press_height
                self._press_height = None
                if dragged and not self._collapsed:
                    self._manual_height = True
                # Also the fit a change during the press was waiting for.
                self._refit_timer.start()
        return super().eventFilter(watched, event)
