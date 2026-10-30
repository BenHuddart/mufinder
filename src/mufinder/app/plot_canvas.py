"""Matplotlib canvases that keep axis labels, ticks and legends visible at any size.

Margins given as figure fractions (``Figure.subplots_adjust``) reserve a fixed
*share* of the canvas, but tick labels and axis labels need a fixed number of
*pixels*. When a dialog opens small (a 13" laptop, or Windows at 125-150 %
display scaling) that share shrinks below what the labels need and they are
clipped, including the minus signs of negative tick values, which silently
flips the sign the reader sees. Constrained layout instead recomputes the
margins from the rendered label extents on every resize.

Every embedded MuFinder plot is created through :func:`create_plot_canvas`, so
each figure uses constrained layout and each canvas has a minimum size below
which a splitter or box layout cannot squeeze it into an unreadable strip.
Plots that share a dialog with other content sit in a :class:`PlotFrame`,
whose corner button opens the plot in its own resizable window.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.icons import icon
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import style_hint_text

DEFAULT_MINIMUM_CANVAS_SIZE: tuple[int, int] = (220, 160)
"""Smallest canvas (logical pixels) that still fits ticks, labels and a plot area."""

POP_OUT_WINDOW_SIZE: tuple[int, int] = (960, 680)
"""Default size of a popped-out plot window, clamped to the screen."""

POP_OUT_WINDOW_FLAGS = (
    Qt.WindowType.Dialog
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowMinMaxButtonsHint
    | Qt.WindowType.WindowCloseButtonHint
)
"""Window type of a popped-out plot: a dialog-type window that keeps its frame buttons.

Several plots live in dialogs run with ``exec()``. While a modal dialog runs,
macOS keeps an ordinary child window (``Qt.Window``) behind the dialog and will
not bring it forward; a dialog-type child is shown above its modal parent and
can be used alongside it. On every platform the window then stays in front of
the dialog that owns the plot instead of disappearing behind it.
"""


def create_plot_canvas(
    *,
    figsize: tuple[float, float],
    facecolor: str | None = None,
    transparent: bool = False,
    minimum_size: tuple[int, int] = DEFAULT_MINIMUM_CANVAS_SIZE,
) -> tuple[Figure, FigureCanvasQTAgg]:
    """Return a constrained-layout figure and the Qt canvas that displays it.

    ``figsize`` only seeds the canvas size hint; the rendered size follows the
    widget. ``transparent`` lets the surrounding Qt surface show through the
    figure background (the figure patch and the canvas widget are both made
    transparent), in which case ``facecolor`` is ignored. ``minimum_size`` is
    the ``(width, height)`` the canvas refuses to shrink below; pass a larger
    value for plots that need more room, never ``(0, 0)``.
    """

    figure = Figure(
        figsize=figsize, facecolor="none" if transparent else facecolor, layout="constrained"
    )
    if transparent:
        figure.patch.set_alpha(0.0)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setMinimumSize(*minimum_size)
    if transparent:
        canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        canvas.setStyleSheet("background: transparent;")
    return figure, canvas


def _leave_navigation_mode(toolbar: object) -> None:
    """Switch a matplotlib navigation toolbar out of pan or zoom mode, releasing its canvas lock."""

    mode = getattr(getattr(toolbar, "mode", None), "name", "")
    if mode == "PAN":
        toolbar.pan()
    elif mode == "ZOOM":
        toolbar.zoom()


class _PlotWindowToolbar(NavigationToolbar2QT):
    """Matplotlib's toolbar without "Configure subplots" and "Edit axes".

    Configuring subplots sets fixed margins, which would undo the constrained
    layout that keeps the labels visible; editing axes is out of scope here.
    """

    toolitems = [
        item for item in NavigationToolbar2QT.toolitems if item[0] not in ("Subplots", "Customize")
    ]


class _PlotWindow(QWidget):
    """Separate resizable window showing a borrowed plot canvas with a navigation toolbar."""

    def __init__(self, canvas: FigureCanvasQTAgg, *, title: str, owner: PlotFrame) -> None:
        super().__init__(owner, POP_OUT_WINDOW_FLAGS)
        self.setObjectName("plotPopOutWindow")
        self.setWindowTitle(title)
        self._owner = owner
        self._canvas = canvas
        # NavigationToolbar2 makes itself the canvas's toolbar; remember the one
        # the dialog installed (e.g. the dendrogram's pan/zoom bar) to put back.
        self._previous_toolbar = canvas.toolbar
        # Only one set of navigation tools drives the canvas at a time: a pan or
        # zoom mode left on in the dialog would hold the canvas's widget lock (so
        # the window's tools would silently do nothing) and its drags would mix
        # with the window's. Switch it off and disable that toolbar until the
        # plot goes back.
        _leave_navigation_mode(self._previous_toolbar)
        self._previous_toolbar_was_enabled = (
            self._previous_toolbar.isEnabled()
            if isinstance(self._previous_toolbar, QWidget)
            else None
        )
        if isinstance(self._previous_toolbar, QWidget):
            self._previous_toolbar.setEnabled(False)
        self._toolbar = _PlotWindowToolbar(canvas, self)
        self._toolbar.setObjectName("plotPopOutToolbar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self._toolbar)
        layout.addWidget(canvas, 1)
        canvas.show()
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)
        self.resize(clamped_size(self, *POP_OUT_WINDOW_SIZE))

    @property
    def toolbar(self) -> NavigationToolbar2QT:
        """The window's navigation toolbar (home, back, forward, pan, zoom, save)."""

        return self._toolbar

    def release_canvas(self) -> FigureCanvasQTAgg:
        """Detach the toolbar from the canvas and hand the canvas back."""

        toolbar = self._toolbar
        _leave_navigation_mode(toolbar)
        for connection in (toolbar._id_press, toolbar._id_release, toolbar._id_drag):
            self._canvas.mpl_disconnect(connection)
        self._canvas.toolbar = self._previous_toolbar
        if (
            isinstance(self._previous_toolbar, QWidget)
            and self._previous_toolbar_was_enabled is not None
        ):
            self._previous_toolbar.setEnabled(self._previous_toolbar_was_enabled)
        self.layout().removeWidget(self._canvas)
        return self._canvas

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().closeEvent(event)
        self._owner.show_here()


class PlotFrame(QWidget):
    """Hold a plot canvas with a button that opens the plot in its own resizable window.

    The window borrows the live canvas rather than copying the figure, so the
    plot keeps updating and stays interactive while it is popped out (and the
    window adds matplotlib's zoom, pan and save toolbar). The frame shows a
    placeholder with a "Show here" button meanwhile, at the canvas's minimum
    size so the dialog layout does not jump. Closing the window, pressing
    Escape in it, or hiding the dialog puts the canvas back.

    Extra controls that belong on the plot (such as the tolerance-scan button)
    can share the frame's top-right overlay through :meth:`add_overlay_widget`.
    """

    popped_out_changed = Signal(bool)

    def __init__(
        self, canvas: FigureCanvasQTAgg, *, title: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("plotFrame")
        self._canvas = canvas
        self._title = title
        self._window: _PlotWindow | None = None
        self._watched_window: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas, 1)

        self._placeholder = QWidget(self)
        self._placeholder.setObjectName("plotPopOutPlaceholder")
        placeholder_layout = QVBoxLayout(self._placeholder)
        placeholder_layout.addStretch(1)
        message = QLabel("Shown in a separate window", self._placeholder)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_hint_text(message)
        placeholder_layout.addWidget(message)
        self._show_here_button = QPushButton("Show here", self._placeholder)
        self._show_here_button.setObjectName("plotShowHereButton")
        self._show_here_button.clicked.connect(self.show_here)
        placeholder_layout.addWidget(self._show_here_button, 0, Qt.AlignmentFlag.AlignHCenter)
        placeholder_layout.addStretch(1)
        self._placeholder.hide()
        layout.addWidget(self._placeholder, 1)

        # Overlaid in the top-right corner rather than given a row of its own,
        # so the button costs the plot no space.
        self._overlay = QWidget(self)
        self._overlay.setObjectName("plotFrameOverlay")
        overlay_layout = QHBoxLayout(self._overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(6)
        # Resize the overlay to fit whatever it currently shows (an overlay
        # control can be hidden), then re-anchor it to the corner.
        overlay_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._overlay.installEventFilter(self)
        self._pop_out_button = QToolButton(self._overlay)
        self._pop_out_button.setObjectName("plotPopOutButton")
        self._pop_out_button.setIcon(icon("pop_out"))
        self._pop_out_button.setAutoRaise(True)
        # Its soft backing, which keeps it visible over plot content, is in the
        # theme stylesheet, so it follows a theme switch with the icon.
        self._pop_out_button.setToolTip("Open in a separate window (zoom, pan, save).")
        self._pop_out_button.setAccessibleName(f"Open {title} in a separate window")
        self._pop_out_button.clicked.connect(self.pop_out)
        overlay_layout.addWidget(self._pop_out_button)
        self._position_overlay()

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        return self._canvas

    @property
    def title(self) -> str:
        return self._title

    @property
    def pop_out_button(self) -> QToolButton:
        return self._pop_out_button

    @property
    def popped_out_window(self) -> QWidget | None:
        return self._window

    def is_popped_out(self) -> bool:
        return self._window is not None

    def add_overlay_widget(self, widget: QWidget) -> None:
        """Show ``widget`` in the top-right overlay, to the left of the pop-out button."""

        layout = self._overlay.layout()
        layout.insertWidget(layout.count() - 1, widget)
        self._position_overlay()

    def pop_out(self) -> None:
        """Move the canvas into its own window, or raise that window if already open."""

        if self._window is not None:
            self._window.raise_()
            self._window.activateWindow()
            return
        self.layout().removeWidget(self._canvas)
        self._placeholder.setMinimumSize(self._canvas.minimumSize())
        self._placeholder.show()
        self._pop_out_button.hide()
        self._position_overlay()
        dialog_title = self.window().windowTitle()
        title = f"{dialog_title} - {self._title}" if dialog_title else self._title
        self._window = _PlotWindow(self._canvas, title=title, owner=self)
        self._watched_window = self.window()
        self._watched_window.installEventFilter(self)
        self._window.show()
        self.popped_out_changed.emit(True)

    def show_here(self) -> None:
        """Put the canvas back in this frame and close its separate window."""

        window = self._window
        if window is None:
            return
        self._window = None
        if self._watched_window is not None:
            self._watched_window.removeEventFilter(self)
            self._watched_window = None
        canvas = window.release_canvas()
        self._placeholder.hide()
        self.layout().insertWidget(0, canvas, 1)
        canvas.show()
        self._pop_out_button.show()
        self._position_overlay()
        window.hide()
        window.deleteLater()
        canvas.draw_idle()
        self.popped_out_changed.emit(False)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        # A separate window is a child of this frame, which Qt does not hide
        # with the dialog; bring the plot home when the dialog goes away.
        if watched is self._watched_window and event.type() in (
            QEvent.Type.Hide,
            QEvent.Type.Close,
        ):
            self.show_here()
        elif watched is self._overlay and event.type() == QEvent.Type.Resize:
            self._position_overlay()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._position_overlay()

    def _position_overlay(self) -> None:
        self._overlay.move(max(self.width() - self._overlay.width() - 4, 0), 4)
        self._overlay.raise_()
