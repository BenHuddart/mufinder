"""Plots embedded beside other content can be opened in their own resizable window."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton, QVBoxLayout
import pytest
import shiboken6

from mufinder.app import theme as theme_module
from mufinder.app.plot_canvas import (
    HOVER_NOTE_OFFSET,
    PlotFrame,
    add_hover_annotation,
    create_plot_canvas,
    hover_annotations,
)
from mufinder.app.theme import plot_badge_box
from tests.support.plots import box_inside, distance_to_box, hover_note_box, panel_positions

sys.path.insert(0, str(Path(__file__).parent))
from test_plot_canvas_layout import PLOT_DIALOGS  # noqa: E402

# Dialogs whose only content is their plot: the dialog itself is already the
# resizable plot window, so they have no pop-out button.
_SINGLE_PLOT_DIALOGS = {"magnetic-field-distribution"}
# Dialogs whose only canvas is a colour key: it has no data to zoom or pan.
_COLOUR_KEY_DIALOGS = {"uep-potential"}


def _settle(qapp, passes: int = 4) -> None:
    for _ in range(passes):
        qapp.processEvents()


class _HostDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Host dialog")
        self.figure, self.canvas = create_plot_canvas(figsize=(4.0, 3.0), facecolor="#ffffff")
        axes = self.figure.add_subplot(111)
        axes.plot([0, 1, 2], [0, 1, 4])
        axes.set_xlabel("x")
        self.previous_toolbar = object()
        self.canvas.toolbar = self.previous_toolbar
        self.frame = PlotFrame(self.canvas, title="Test plot", parent=self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.frame)
        self.resize(640, 480)


@pytest.fixture
def host(qapp):
    dialog = _HostDialog()
    dialog.show()
    _settle(qapp)
    yield dialog
    dialog.frame.show_here()
    dialog.close()


def _callback_counts(canvas: FigureCanvasQTAgg) -> dict[str, int]:
    return {name: len(callbacks) for name, callbacks in canvas.callbacks.callbacks.items()}


def test_pop_out_moves_the_live_canvas_into_its_own_window(qapp, host) -> None:
    host.frame.pop_out_button.click()
    _settle(qapp)

    window = host.frame.popped_out_window
    assert host.frame.is_popped_out()
    assert window is not None and window.isVisible()
    assert window.isWindow()
    assert host.canvas.window() is window
    assert window.windowTitle() == "Host dialog - Test plot"
    # Borrowed, not copied: the dialog's figure and canvas are the ones shown.
    assert host.canvas.figure is host.figure
    assert window.width() > host.frame.width()
    # The frame keeps its size and offers to bring the plot back.
    placeholder = host.frame.findChild(QPushButton, "plotShowHereButton")
    assert placeholder is not None and placeholder.isVisible()
    assert placeholder.parentWidget().minimumSize() == host.canvas.minimumSize()
    assert host.frame.pop_out_button.isHidden()


def test_pop_out_window_is_a_resizable_dialog_type_window(qapp, host) -> None:
    """A plain ``Qt.Window`` child stays stuck behind a modal dialog on macOS.

    The relaxation clustering and displacement dialogs run with ``exec()``; a
    dialog-type child window is shown above its modal parent and can be used.
    """

    host.frame.pop_out()
    window = host.frame.popped_out_window

    assert window.windowType() == Qt.WindowType.Dialog
    flags = window.windowFlags()
    for hint in (
        Qt.WindowType.WindowMinimizeButtonHint,
        Qt.WindowType.WindowMaximizeButtonHint,
        Qt.WindowType.WindowCloseButtonHint,
    ):
        assert flags & hint, hint
    assert window.windowModality() == Qt.WindowModality.NonModal


def test_pop_out_window_can_be_used_while_its_dialog_runs_modally(qapp) -> None:
    dialog = _HostDialog()
    seen = []

    def interact() -> None:
        dialog.frame.pop_out()
        window = dialog.frame.popped_out_window
        # Qt does not block a window whose (transient) parent is the modal dialog.
        seen.append(
            (
                window.isVisible(),
                window.windowType(),
                QApplication.modalWindow() is dialog.windowHandle(),
            )
        )
        window.toolbar.zoom()
        seen.append(dialog.canvas.toolbar is window.toolbar)
        window.close()
        dialog.reject()

    QTimer.singleShot(0, interact)
    dialog.exec()

    assert seen == [(True, Qt.WindowType.Dialog, True), True]
    assert not dialog.frame.is_popped_out()
    assert dialog.canvas.widgetlock.available(None)


def test_pop_out_window_offers_navigation_without_layout_editing(qapp, host) -> None:
    host.frame.pop_out()
    _settle(qapp)

    toolbar = host.frame.popped_out_window.toolbar
    actions = {action.text() for action in toolbar.actions() if action.text()}
    assert {"Home", "Back", "Forward", "Pan", "Zoom", "Save"} <= actions
    # Configuring subplots would replace the constrained layout with fixed margins.
    assert not actions & {"Subplots", "Customize"}
    assert host.canvas.toolbar is toolbar


def test_pop_out_takes_over_from_an_active_dialog_toolbar(qapp) -> None:
    """A dialog toolbar left in pan mode must not lock the window's tools out."""

    dialog = QDialog()
    figure, canvas = create_plot_canvas(figsize=(4.0, 3.0), facecolor="#ffffff")
    figure.add_subplot(111).plot([0, 1], [0, 1])
    layout = QVBoxLayout(dialog)
    dialog_toolbar = NavigationToolbar2QT(canvas, dialog)
    layout.addWidget(dialog_toolbar)
    frame = PlotFrame(canvas, title="Test plot", parent=dialog)
    layout.addWidget(frame)
    dialog.show()
    _settle(qapp)
    dialog_toolbar.pan()
    assert not canvas.widgetlock.available(None)

    frame.pop_out()
    window = frame.popped_out_window

    assert dialog_toolbar.mode.name == "NONE"
    assert not dialog_toolbar.isEnabled()
    window.toolbar.zoom()
    assert window.toolbar.mode.name == "ZOOM"

    frame.show_here()
    assert canvas.toolbar is dialog_toolbar
    assert dialog_toolbar.isEnabled()
    assert canvas.widgetlock.available(None)
    dialog_toolbar.pan()
    assert dialog_toolbar.mode.name == "PAN"
    dialog_toolbar.pan()
    dialog.close()


def test_popping_out_twice_raises_the_same_window(qapp, host) -> None:
    host.frame.pop_out()
    first = host.frame.popped_out_window
    host.frame.pop_out()

    assert host.frame.popped_out_window is first


@pytest.mark.parametrize("how", ["show-here", "close-window", "hide-dialog"])
def test_plot_returns_to_the_frame_with_its_toolbar_detached(qapp, host, how: str) -> None:
    callbacks_before = _callback_counts(host.canvas)
    host.frame.pop_out()
    _settle(qapp)
    window = host.frame.popped_out_window
    window.toolbar.pan()
    assert not host.canvas.widgetlock.available(None)

    if how == "show-here":
        host.frame.findChild(QPushButton, "plotShowHereButton").click()
    elif how == "close-window":
        window.close()
    else:
        host.hide()
    _settle(qapp)

    assert not host.frame.is_popped_out()
    assert host.canvas.parentWidget() is host.frame
    # The window is hidden and then deleted once the plot is back.
    assert not shiboken6.isValid(window) or not window.isVisible()
    assert host.canvas.toolbar is host.previous_toolbar
    assert host.canvas.widgetlock.available(None)
    assert _callback_counts(host.canvas) == callbacks_before
    if how != "hide-dialog":
        assert host.canvas.isVisible()
        assert not host.frame.pop_out_button.isHidden()


def test_escape_closes_the_pop_out_window(qapp, host) -> None:
    host.frame.pop_out()
    _settle(qapp)
    window = host.frame.popped_out_window
    shortcuts = [
        child for child in window.children() if child.metaObject().className() == "QShortcut"
    ]

    assert any(shortcut.key().toString() == "Esc" for shortcut in shortcuts)
    next(shortcut for shortcut in shortcuts if shortcut.key().toString() == "Esc").activated.emit()
    _settle(qapp)
    assert not host.frame.is_popped_out()


def test_overlay_widgets_sit_left_of_the_pop_out_button_in_the_top_right_corner(qapp, host) -> None:
    extra = QPushButton("Scan")
    host.frame.add_overlay_widget(extra)
    host.resize(700, 500)
    _settle(qapp)

    button = host.frame.pop_out_button
    button_right = button.mapTo(host.frame, QPoint(button.width(), 0)).x()
    assert host.frame.width() - button_right <= 6
    assert button.mapTo(host.frame, QPoint(0, 0)).y() <= 6
    assert (
        extra.mapTo(host.frame, QPoint(extra.width(), 0)).x()
        <= button.mapTo(host.frame, QPoint(0, 0)).x()
    )

    # Hiding an overlay control shrinks the overlay; the button stays in the corner.
    extra.hide()
    _settle(qapp)
    button_right = button.mapTo(host.frame, QPoint(button.width(), 0)).x()
    assert host.frame.width() - button_right <= 6


@pytest.mark.parametrize(
    "name", sorted(set(PLOT_DIALOGS) - _SINGLE_PLOT_DIALOGS - _COLOUR_KEY_DIALOGS)
)
def test_plots_sharing_a_dialog_can_be_popped_out(qapp, name: str) -> None:
    dialog = PLOT_DIALOGS[name]()
    try:
        dialog.show()
        dialog.resize(1240, 740)
        _settle(qapp)
        canvases = [
            canvas
            for canvas in dialog.findChildren(FigureCanvasQTAgg)
            if canvas.isVisibleTo(dialog) and canvas.figure.axes
        ]
        assert canvases
        for canvas in canvases:
            frame = canvas.parentWidget()
            assert isinstance(frame, PlotFrame), f"{name}: a plot has no pop-out button"
            assert frame.pop_out_button.isVisibleTo(dialog)
            assert frame.title
    finally:
        dialog.close()


def test_pinned_convergence_plots_can_be_popped_out(qapp) -> None:
    from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget

    overview = ConvergenceOverviewWidget()
    for card in (overview._cutoff_card, overview._kpoint_card):
        assert isinstance(card._plot._canvas.parentWidget(), PlotFrame)
    overview.close()


def test_pinned_convergence_placeholder_brings_a_popped_out_plot_back(qapp) -> None:
    from mufinder.app.convergence_overview_widget import _PinnedPlotWidget

    plot = _PinnedPlotWidget(sweep_kind="cutoff")
    plot._canvas_frame.show()
    plot._canvas.show()
    plot.show()
    _settle(qapp)
    plot._canvas_frame.pop_out()
    _settle(qapp)
    assert plot._canvas_frame.is_popped_out()

    plot.show_placeholder("Nothing pinned")
    _settle(qapp)

    assert not plot._canvas_frame.is_popped_out()
    assert plot._canvas_frame.isHidden()
    assert plot._canvas.isHidden()
    plot.close()


def test_pop_out_button_backing_follows_a_theme_switch(qapp, host) -> None:
    # The backing comes from the theme stylesheet, like the button's icon from
    # the palette, so both change together and the icon keeps its contrast.
    def backing_colour() -> QColor:
        image = host.frame.pop_out_button.grab().toImage()
        return image.pixelColor(image.width() // 2, 3)

    try:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        _settle(qapp)
        assert backing_colour().name() == theme_module.light_palette().bg_surface.lower()

        theme_module.apply_theme(qapp, theme_module.THEME_DARK)
        _settle(qapp)
        assert backing_colour().name() == theme_module.dark_palette().bg_surface.lower()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def _hover_over(canvas: FigureCanvasQTAgg, axes, x: float, y: float) -> None:
    """Move the pointer over data point ``(x, y)`` of ``axes`` as a real mouse move does."""

    pixel_x, pixel_y = axes.transData.transform((x, y))
    MouseEvent("motion_notify_event", canvas, pixel_x, pixel_y)._process()


def _save_through_toolbar(monkeypatch, canvas, toolbar, path: Path | None) -> list[bool]:
    """Press the toolbar's save tool, answering its file dialog with ``path`` (``None`` cancels).

    Returns, for each figure render made while the save runs, whether any hover
    annotation was visible in it.
    """

    monkeypatch.setitem(mpl.rcParams, "savefig.directory", str(path.parent if path else ""))
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(path) if path else "", ""),
    )
    renders: list[bool] = []
    connection = canvas.mpl_connect(
        "draw_event",
        lambda _event: renders.append(
            any(artist.get_visible() for artist in hover_annotations(canvas.figure))
        ),
    )
    try:
        save_action = next(action for action in toolbar.actions() if action.text() == "Save")
        save_action.trigger()
    finally:
        canvas.mpl_disconnect(connection)
    return renders


def _screen_draws(qapp, canvas) -> list[bool]:
    """Let pending events run; return, per screen draw, whether a hover annotation showed."""

    draws: list[bool] = []
    connection = canvas.mpl_connect(
        "draw_event",
        lambda _event: draws.append(
            any(artist.get_visible() for artist in hover_annotations(canvas.figure))
        ),
    )
    try:
        _settle(qapp)
    finally:
        canvas.mpl_disconnect(connection)
    return draws


def test_toolbar_save_leaves_out_a_shown_hover_annotation(qapp, host, monkeypatch, tmp_path):
    annotation = add_hover_annotation(host.figure.axes[0], "Point 1", xy=(1.0, 1.0))
    host.frame.pop_out()
    _settle(qapp)
    host.canvas.draw()

    saved = tmp_path / "plot.png"
    renders = _save_through_toolbar(
        monkeypatch, host.canvas, host.frame.popped_out_window.toolbar, saved
    )

    assert saved.is_file()
    assert renders and not any(renders)
    # Still pointed at, the annotation is back on screen once the save is done.
    assert annotation.get_visible()
    screen = _screen_draws(qapp, host.canvas)
    assert screen and all(screen)


def test_cancelled_toolbar_save_keeps_the_hover_annotation_without_redrawing(
    qapp, host, monkeypatch
):
    annotation = add_hover_annotation(host.figure.axes[0], "Point 1", xy=(1.0, 1.0))
    host.frame.pop_out()
    _settle(qapp)
    host.canvas.draw()

    renders = _save_through_toolbar(
        monkeypatch, host.canvas, host.frame.popped_out_window.toolbar, None
    )

    assert renders == []
    assert annotation.get_visible()
    assert _screen_draws(qapp, host.canvas) == []


@pytest.fixture
def plain_canvas(qapp):
    """A shown plot canvas with matplotlib's own toolbar, outside any frame."""

    figure, canvas = create_plot_canvas(figsize=(4.0, 3.0), facecolor="#ffffff")
    figure.add_subplot(111).plot([0, 1, 2], [0, 1, 4])
    canvas.resize(640, 480)
    canvas.show()
    _settle(qapp)
    canvas.draw()
    _settle(qapp)
    yield canvas
    canvas.close()


def test_hover_annotation_does_not_change_the_saved_image(qapp, plain_canvas, tmp_path):
    figure = plain_canvas.figure
    before = tmp_path / "before.png"
    figure.savefig(before)
    # Near the top-right corner, where a shown note would push the layout in.
    annotation = add_hover_annotation(figure.axes[0], "A long note near the corner", xy=(2.0, 4.0))
    plain_canvas.draw()

    after = tmp_path / "after.png"
    figure.savefig(after)

    assert annotation.get_visible()
    assert after.read_bytes() == before.read_bytes()


def test_convergence_examiner_save_leaves_out_its_hover_tooltip(qapp, monkeypatch, tmp_path):
    dialog = PLOT_DIALOGS["convergence-examiner"]()
    try:
        dialog.show()
        _settle(qapp)
        dialog._run_analysis()
        frame = next(iter(dialog.findChildren(PlotFrame)))
        frame.pop_out()
        _settle(qapp)
        canvas = frame.canvas
        canvas.draw()
        energy_axes = canvas.figure.axes[0]
        x_value = energy_axes.lines[0].get_xdata()[0]
        y_value = energy_axes.lines[0].get_ydata()[0]
        _hover_over(canvas, energy_axes, float(x_value), float(y_value))
        shown = [artist for artist in hover_annotations(canvas.figure) if artist.get_visible()]
        assert shown, "hovering a point shows no hover annotation"

        saved = tmp_path / "examiner.png"
        renders = _save_through_toolbar(monkeypatch, canvas, frame.popped_out_window.toolbar, saved)

        assert saved.is_file()
        assert renders and not any(renders)
        assert all(artist.get_visible() for artist in shown)
        screen = _screen_draws(qapp, canvas)
        assert screen and all(screen)
    finally:
        dialog.close()


def _place_note(canvas, at, text="Cutoff: 500 eV\nEnergy: -10.0105 eV"):
    """Show a hover note on a point at ``at``, a share of its panel's width and height.

    Returns the note, its drawn box and the point, in display pixels.
    """

    axes = canvas.figure.axes[0]
    for shown in hover_annotations(canvas.figure):
        shown.remove()
    canvas.draw()
    xy = tuple(axes.transData.inverted().transform(axes.transAxes.transform(at)))
    note = add_hover_annotation(axes, text, xy=xy, bbox=plot_badge_box(pad=0.3), fontsize=9)
    canvas.draw()
    return note, hover_note_box(note, canvas.get_renderer()), axes.transData.transform(xy)


def test_hover_note_that_fits_sits_where_a_plain_annotation_does(qapp, plain_canvas) -> None:
    axes = plain_canvas.figure.axes[0]
    style = {"bbox": plot_badge_box(pad=0.3), "fontsize": 9}
    plain = axes.annotate(
        "Cutoff: 500 eV", xy=(0.5, 1.0), xytext=(10, 10), textcoords="offset points", **style
    )
    note = add_hover_annotation(axes, "Cutoff: 500 eV", xy=(0.5, 1.0), **style)
    plain_canvas.draw()
    renderer = plain_canvas.get_renderer()

    assert tuple(note.get_position()) == HOVER_NOTE_OFFSET == (10.0, 10.0)
    assert np.allclose(
        hover_note_box(note, renderer).extents, hover_note_box(plain, renderer).extents
    )


@pytest.mark.parametrize(
    ("at", "flipped_left", "flipped_down"),
    [
        ((0.99, 0.5), True, False),
        ((0.3, 0.99), False, True),
        ((0.99, 0.99), True, True),
    ],
    ids=["right-edge", "top-edge", "top-right-corner"],
)
def test_hover_note_flips_to_the_other_side_of_its_point_at_the_same_gap(
    qapp, plain_canvas, at, flipped_left, flipped_down
) -> None:
    _note, preferred, point = _place_note(plain_canvas, (0.3, 0.3))
    gap_x, gap_y = preferred.x0 - point[0], preferred.y0 - point[1]
    assert gap_x > 0 and gap_y > 0

    _note, box, point = _place_note(plain_canvas, at)

    assert box_inside(box, plain_canvas.figure.bbox)
    if flipped_left:
        assert box.x1 == pytest.approx(point[0] - gap_x, abs=0.5)
    else:
        assert box.x0 == pytest.approx(point[0] + gap_x, abs=0.5)
    if flipped_down:
        assert box.y1 == pytest.approx(point[1] - gap_y, abs=0.5)
    else:
        assert box.y0 == pytest.approx(point[1] + gap_y, abs=0.5)


def test_hover_note_with_no_room_either_side_slides_along_until_it_fits(qapp, plain_canvas) -> None:
    long_text = "A hover note that is far wider than the room on either side of its point"
    _note, preferred, preferred_point = _place_note(plain_canvas, (0.01, 0.3), text=long_text)
    _note, box, point = _place_note(plain_canvas, (0.5, 0.5), text=long_text)
    figure = plain_canvas.figure.bbox
    assert box.width > max(point[0] - figure.x0, figure.x1 - point[0])

    assert box_inside(box, figure)
    # Still above the point at the usual gap, so the point stays clear.
    assert box.y0 == pytest.approx(point[1] + preferred.y0 - preferred_point[1], abs=0.5)
    assert box.y0 > point[1]


def test_hover_note_larger_than_the_figure_keeps_its_first_line_in_view(qapp, plain_canvas) -> None:
    lines = [
        f"Line {number}: " + "a hover note far wider than the figure " * 4 for number in range(60)
    ]
    _note, box, _point = _place_note(plain_canvas, (0.5, 0.5), text="\n".join(lines))
    figure = plain_canvas.figure.bbox
    assert box.height > figure.height and box.width > figure.width

    assert box.y1 == pytest.approx(figure.y1, abs=0.5)
    assert box.x0 == pytest.approx(figure.x0, abs=0.5)


def test_hover_note_may_cover_a_neighbouring_panel_to_stay_beside_its_point(qapp) -> None:
    figure, canvas = create_plot_canvas(figsize=(4.0, 3.0), facecolor="#ffffff")
    upper, lower = figure.subplots(2, 1)
    upper.plot([0, 1, 2], [0, 1, 4])
    lower.plot([0, 1, 2], [0, 1, 4])
    canvas.resize(640, 480)
    canvas.show()
    _settle(qapp)
    try:
        # A point at the top of the lower panel, just below the upper panel.
        xy = (1.0, lower.get_ylim()[1] - 0.01 * np.ptp(lower.get_ylim()))
        note = add_hover_annotation(lower, "Point 3\nValue: 4", xy=xy, bbox=plot_badge_box())
        canvas.draw()
        box = hover_note_box(note, canvas.get_renderer())
        point = lower.transData.transform(xy)

        assert box.y0 > point[1]
        assert box.overlaps(upper.bbox)
        assert box_inside(box, figure.bbox)
        assert distance_to_box(box, *point) <= math.hypot(*HOVER_NOTE_OFFSET) * figure.dpi / 72
    finally:
        canvas.close()


def test_hover_note_is_drawn_above_every_panel_and_legend(qapp, plain_canvas, monkeypatch) -> None:
    figure = plain_canvas.figure
    axes = figure.axes[0]
    axes.plot([0, 2], [4, 0], label="Second line")
    legend = axes.legend(loc="upper left")
    twin = axes.twinx()
    twin.plot([0, 2], [1, 3], marker="o")
    note = add_hover_annotation(axes, "Point 1", xy=(0.0, 4.0))
    drawn: list[object] = []
    for artist in (note, legend, axes, twin):
        original = artist.draw
        monkeypatch.setattr(
            artist,
            "draw",
            lambda renderer, _artist=artist, _draw=original: (
                drawn.append(_artist),
                _draw(renderer),
            ),
        )

    plain_canvas.draw()

    assert drawn[-1] is note
    assert {id(artist) for artist in drawn} == {id(note), id(legend), id(axes), id(twin)}


def test_hover_note_moves_no_panel(qapp, plain_canvas) -> None:
    figure = plain_canvas.figure
    plain_canvas.draw()
    before = panel_positions(figure)

    note = add_hover_annotation(
        figure.axes[0], "A long note near the corner of the plot", xy=(2.0, 4.0)
    )
    plain_canvas.draw()

    assert note.get_visible()
    assert panel_positions(figure) == pytest.approx(before, abs=1e-12)


def test_hover_note_is_not_drawn_while_its_point_is_outside_the_panel(qapp, plain_canvas) -> None:
    figure = plain_canvas.figure
    axes = figure.axes[0]
    note = add_hover_annotation(axes, "Point 3", xy=(2.0, 4.0), bbox=plot_badge_box())
    plain_canvas.draw()
    assert note.get_window_extent().width > 0
    drawn = np.array(plain_canvas.buffer_rgba())

    # Zoomed in, as the pop-out window's tools do, so the point is off the panel.
    axes.set_xlim(0.0, 1.0)
    plain_canvas.draw()
    zoomed = np.array(plain_canvas.buffer_rgba())
    note.set_visible(False)
    plain_canvas.draw()

    assert np.array_equal(zoomed, np.array(plain_canvas.buffer_rgba()))
    assert not np.array_equal(drawn, zoomed)


@pytest.mark.parametrize(
    "keyword", ["xytext", "textcoords", "xycoords", "annotation_clip", "zorder", "in_layout"]
)
def test_hover_note_refuses_a_placement_of_its_own(qapp, plain_canvas, keyword) -> None:
    with pytest.raises(TypeError, match=keyword):
        add_hover_annotation(plain_canvas.figure.axes[0], "Point", xy=(1.0, 1.0), **{keyword: 5})
