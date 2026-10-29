"""Plots embedded beside other content can be opened in their own resizable window."""

from __future__ import annotations

from pathlib import Path
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout
import pytest
import shiboken6

from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas

sys.path.insert(0, str(Path(__file__).parent))
from test_plot_canvas_layout import PLOT_DIALOGS  # noqa: E402

# Dialogs whose only content is their plot: the dialog itself is already the
# resizable plot window, so they have no pop-out button.
_SINGLE_PLOT_DIALOGS = {"magnetic-field-distribution", "convergence-examiner"}


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


@pytest.mark.parametrize("name", sorted(set(PLOT_DIALOGS) - _SINGLE_PLOT_DIALOGS))
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


def test_quantum_plot_and_pinned_convergence_plots_can_be_popped_out(qapp, tmp_path) -> None:
    import test_quantum_analysis_dialog as quantum_tests

    from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget
    from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog

    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(quantum_tests.PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), job_label="quantum")
    quantum_tests._show_tab(dialog, "Mode wells and widths")
    assert isinstance(dialog._canvas.parentWidget(), PlotFrame)
    dialog.close()

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
