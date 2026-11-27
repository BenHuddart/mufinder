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

A note that follows the pointer over a plot is created with
:func:`add_hover_annotation`, which places it beside its point and whole inside
the figure, above every panel and legend. The canvas leaves every such note out of an
image saved from it, whichever toolbar or path saves it, so a saved image
shows the plot only. :func:`nearest_point_on_screen` finds the point such a
note belongs to, measuring in pixels so that a log axis behaves like a linear one.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
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
from mufinder.app.theme import style_hint_text, style_transparent_surface

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


_HOVER_ANNOTATION_MARK = "_mufinder_hover_annotation"

HOVER_NOTE_OFFSET: tuple[float, float] = (10.0, 10.0)
"""Where a hover note's text starts, in points right of and above its point."""

HOVER_NOTE_ZORDER = 1000.0
"""Drawing order of a hover note: after every panel, legend and other figure artist."""

_HOVER_NOTE_PLACEMENT_KEYWORDS = frozenset(
    {"xytext", "textcoords", "xycoords", "annotation_clip", "zorder", "in_layout"}
)


class _HoverNote(Annotation):
    """A note beside a data point that stays whole inside its figure.

    The note is a figure artist, so it is drawn above every panel and legend,
    and it takes no part in the layout. Each draw places it by one rule (see
    :func:`add_hover_annotation`) from the figure's current size. Like any
    annotation of a data point, it is not drawn while its point is outside its
    panel (a zoomed-in view, say).
    """

    def draw(self, renderer) -> None:
        if self.get_visible():
            self._place(renderer)
        super().draw(renderer)

    def _drawn_box(self, renderer) -> Bbox:
        """The note's extent on screen, its box and the box's border included, in display pixels."""

        self.update_positions(renderer)
        self.update_bbox_position_size(renderer)
        boxes = [Text.get_window_extent(self, renderer)]
        patch = self.get_bbox_patch()
        if patch is not None:
            # The patch's extent is its outline's path; half the border lies outside it.
            border = patch.get_linewidth() * self.figure.dpi / 72.0 / 2.0
            boxes.append(patch.get_window_extent(renderer).padded(border))
        return Bbox.union(boxes)

    def _place(self, renderer) -> None:
        figure = self.figure
        if figure is None or self.axes is None:
            return
        self.xyann = HOVER_NOTE_OFFSET
        box = self._drawn_box(renderer)
        point_x, point_y = self.axes.transData.transform(self.xy)
        dx, dy = _hover_note_shift(box, figure.bbox, point_x, point_y)
        if dx or dy:
            # A shift in display pixels, as an offset in points from the point.
            scale = 72.0 / figure.dpi
            self.xyann = (HOVER_NOTE_OFFSET[0] + dx * scale, HOVER_NOTE_OFFSET[1] + dy * scale)


def _hover_note_shift(
    box: Bbox, bounds: Bbox, point_x: float, point_y: float
) -> tuple[float, float]:
    """Return how far to move a note from its preferred place, up and right of its point.

    On each axis the note keeps its preferred side while it fits within
    ``bounds``; otherwise it flips to the opposite side of the point at the same
    gap, and if it still does not fit it slides along until it does. A note
    larger than the bounds keeps its first line's start in view: its left edge
    and its top edge. All values are display pixels.
    """

    def along(
        low: float, high: float, point: float, lower: float, upper: float, *, keep_high: bool
    ) -> float:
        shift = 0.0
        if high > upper:
            gap = low - point
            shift = (point - gap) - high
        # Slide inside the bounds; the edge to keep in view is clamped last.
        if keep_high:
            return min(max(shift, lower - low), upper - high)
        return max(min(shift, upper - high), lower - low)

    return (
        along(box.x0, box.x1, point_x, bounds.x0, bounds.x1, keep_high=False),
        along(box.y0, box.y1, point_y, bounds.y0, bounds.y1, keep_high=True),
    )


def add_hover_annotation(
    axes: Axes, text: str, *, xy: tuple[float, float], **kwargs: object
) -> Annotation:
    """Annotate the data point ``xy`` of ``axes`` with a note that follows the pointer.

    The note sits beside its point with a small fixed gap, its text starting
    :data:`HOVER_NOTE_OFFSET` points right of and above it. When there is no
    room on that side within the figure, it flips to the opposite side of the
    point; if it still does not fit, it slides along that side until it does.
    The boundary is the whole figure, not the note's panel: the note may cover a
    neighbouring panel, and it is drawn above everything, legends included. It
    takes no part in the layout, so showing it never moves a panel.

    Other keyword arguments style the text, as for
    :meth:`matplotlib.axes.Axes.annotate`; where the note goes is this rule's
    alone, so ``xytext``, ``textcoords``, ``xycoords`` and ``annotation_clip``
    are refused, as are ``zorder`` and ``in_layout``. The note is not drawn
    while its point is outside the panel, and it is left out of any image saved
    from a canvas made by :func:`create_plot_canvas`.
    """

    placement = _HOVER_NOTE_PLACEMENT_KEYWORDS.intersection(kwargs)
    if placement:
        raise TypeError(f"add_hover_annotation places the note itself: {sorted(placement)}")
    annotation = _HoverNote(
        text,
        xy=xy,
        xycoords="data",
        xytext=HOVER_NOTE_OFFSET,
        textcoords="offset points",
        zorder=HOVER_NOTE_ZORDER,
        in_layout=False,
        **kwargs,
    )
    annotation.axes = axes
    axes.figure.add_artist(annotation)
    setattr(annotation, _HOVER_ANNOTATION_MARK, True)
    return annotation


def nearest_point_on_screen(
    axes: Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    event: MouseEvent,
    *,
    max_distance: float | None = None,
) -> int | None:
    """Return the index of the point of ``axes`` horizontally nearest ``event`` on screen.

    Distances are measured in display pixels (device pixels, as matplotlib's
    events count them) through the axes' own scales, so a point on a log axis
    is as near as it looks. ``max_distance`` (device pixels) leaves out a point
    farther than that; ``None`` is returned when no point is near enough.
    """

    if not len(x_values) or event.x is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        points = axes.transData.transform(
            np.column_stack((np.asarray(x_values, float), np.asarray(y_values, float)))
        )
    distances = np.nan_to_num(np.abs(points[:, 0] - event.x), nan=np.inf)
    nearest = int(np.argmin(distances))
    if not np.isfinite(distances[nearest]):
        return None
    if max_distance is not None and distances[nearest] > max_distance:
        return None
    return nearest


def hover_annotations(figure: Figure) -> list[Artist]:
    """Return every annotation on ``figure`` made by :func:`add_hover_annotation`."""

    return figure.findobj(lambda artist: getattr(artist, _HOVER_ANNOTATION_MARK, False))


class _PlotCanvas(FigureCanvasQTAgg):
    """Qt canvas that saves its figure without the notes that follow the pointer."""

    def print_figure(self, *args, **kwargs):
        # Every save goes through here (the toolbar's save tool and
        # ``Figure.savefig``).
        shown = [artist for artist in hover_annotations(self.figure) if artist.get_visible()]
        for artist in shown:
            artist.set_visible(False)
        try:
            return super().print_figure(*args, **kwargs)
        finally:
            for artist in shown:
                artist.set_visible(True)
            if shown:
                # Saving a raster image renders into the canvas's own buffer;
                # draw the notes back for the pointer that is still there.
                self.draw_idle()


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
    canvas = _PlotCanvas(figure)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setMinimumSize(*minimum_size)
    if transparent:
        canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        style_transparent_surface(canvas)
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
